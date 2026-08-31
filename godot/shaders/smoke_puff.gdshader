shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_disabled, unshaded;

// Volumetric ~air molecule~ mist.
//
// Every particle of the Rust air pool is a small unit quad, billboarded in
// the vertex shader toward the camera (same camera-plane projection the old
// streak shader used, extended to a pure radial billboard). The fragment
// shader scores each quad as a soft round seed whose density is mottled with
// world-space value noise and cross-faded with downstream-traveling shimmer
// bands so the air keeps visibly streaming even where the advection field is
// near-uniform. Age (0 fresh at the rake -> 1 dissipating) widens and fades.
//
// Per-instance custom data:  x = age_norm, y = per-particle seed [0,1).

uniform vec3 base_color : source_color = vec3(0.95, 0.99, 1.0);
uniform vec3 aged_color : source_color = vec3(0.60, 0.80, 0.97);
uniform float core_alpha : hint_range(0.0, 1.0) = 0.85;
uniform float edge_soft : hint_range(0.0, 1.0) = 0.25;
uniform float shine : hint_range(0.0, 1.0) = 0.6;
uniform float noise_scale : hint_range(0.05, 2.0) = 0.55;
uniform float noise_speed : hint_range(0.0, 2.0) = 0.8;
uniform float flow_speed : hint_range(0.0, 200.0) = 20.0;
uniform float glow : hint_range(0.0, 2.0) = 1.2;
uniform float age_fade : hint_range(0.0, 1.0) = 0.45;

varying flat float v_age;
varying flat float v_seed;

// --- Smooth value noise (from the working streak shader) --------------------
float hash21(vec3 p) {
	p = fract(p * 0.1031);
	p += dot(p, p.yzx + 33.33);
	return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	vec3 u = f * f * (3.0 - 2.0 * f);
	float a = hash21(i);
	float b = hash21(i + vec3(1.0, 0.0, 0.0));
	float c = hash21(i + vec3(0.0, 1.0, 0.0));
	float d = hash21(i + vec3(1.0, 1.0, 0.0));
	float e = hash21(i + vec3(0.0, 0.0, 1.0));
	float f0 = hash21(i + vec3(1.0, 0.0, 1.0));
	float g = hash21(i + vec3(0.0, 1.0, 1.0));
	float h = hash21(i + vec3(1.0, 1.0, 1.0));
	return mix(
		mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
		mix(mix(e, f0, u.x), mix(g, h, u.x), u.y),
		u.z);
}

void vertex() {
	// Billboarding: quad center in world space, then vertex offset mapped onto
	// the camera plane (columns 0/1 of the view matrix = world right/up axes).
	vec3 origin = (MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec3 cam_right = vec3(VIEW_MATRIX[0][0], VIEW_MATRIX[1][0], VIEW_MATRIX[2][0]);
	vec3 cam_up = vec3(VIEW_MATRIX[0][1], VIEW_MATRIX[1][1], VIEW_MATRIX[2][1]);

	// Instance scale is (r, r, 1), so column 0 = the radius (metres).
	float r = length(MODEL_MATRIX[0].xyz);
	vec3 offset = (cam_right * VERTEX.x + cam_up * VERTEX.y) * 2.0 * r;

	POSITION = PROJECTION_MATRIX * (VIEW_MATRIX * vec4(origin + offset, 1.0));
	v_age = INSTANCE_CUSTOM.x;
	v_seed = INSTANCE_CUSTOM.y;
}

void fragment() {
	// Round, soft seed profile across the billboard.
	float d = length(VERTEX.xy) * 2.0;
	float prof = 1.0 - smoothstep(1.0 - edge_soft, 1.0, d);
	prof *= prof; // rounder falloff

	// World position of the pixel (VERTEX is view-space here).
	vec3 wp = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;

	// Streaming motion: downstream-traveling shimmer bands anchored in the LAB
	// frame, riding at the live tunnel wind. Each seed shifts its phase so
	// adjacent molecules shimmer independently instead of blinking in lockstep.
	float s = -wp.x;
	float w1 = sin(s * 1.3 - TIME * min(flow_speed * 1.3, 45.0) + v_seed * 41.7);
	float w2 = sin(s * 3.1 - TIME * min(flow_speed * 2.6, 70.0) + v_seed * 93.3);
	float travel = 0.5 + 0.5 * (0.65 * w1 + 0.35 * w2);
	travel = pow(travel, 1.4);

	// Molecular mottling: slow-evolving world-space noise, seeded per particle.
	vec3 bp = wp * noise_scale + v_seed * 37.0;
	bp.z += TIME * noise_speed;
	float n0 = vnoise(bp);
	float n1 = vnoise(bp * 2.1 + 1.7);
	float mottle = 0.45 + 0.55 * (n0 * 0.65 + n1 * 0.35);

	float brightness = mix(mottle, travel, shine);

	float age = clamp(v_age, 0.0, 1.0);
	float alpha = prof * core_alpha * brightness;
	alpha *= 1.0 - age_fade * age * age;

	vec3 col = mix(base_color, aged_color, age * 0.55);

	ALPHA = clamp(alpha, 0.0, 1.0);
	ALBEDO = col;
	EMISSION = col * (0.35 + alpha) * glow;
}