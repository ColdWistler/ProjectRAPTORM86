// Dense volumetric wind-tunnel smoke.
//
// Renders GPUParticles3D instances as soft, overlapping billboarded puffs that
// read as dense cloud smoke.  Alpha blending is used so overlapping puffs
// accumulate opacity; a broad soft profile + additive glow makes the aggregate
// read as a continuous volume instead of a field of hard dots.
//
// The particle pool is dense (~25k) and each particle is a wide soft puff, so
// the overlapping sprites build up a thick, volumetric-looking smoke layer.
//
// Per-instance COLOR (packed by wind_tunnel_flow.gdshader):
//   R = age_norm (0 fresh -> 1 dissipating)
//   G,B = 24-bit seed

shader_type spatial;
render_mode blend_add, depth_draw_never, cull_disabled, unshaded, disable_fog;

// --- Tuning -----------------------------------------------------------------
uniform vec3 base_color : source_color = vec3(0.92, 0.97, 1.0);
uniform vec3 aged_color : source_color = vec3(0.55, 0.75, 0.92);
uniform float core_alpha : hint_range(0.05, 1.0) = 0.18;
uniform float puff_soft : hint_range(0.0, 1.0) = 0.65;
uniform float puff_scale : hint_range(0.5, 4.0) = 2.4;
uniform float noise_scale : hint_range(0.05, 3.0) = 0.45;
uniform float noise_speed : hint_range(0.0, 3.0) = 0.6;
uniform float flow_speed : hint_range(0.0, 200.0) = 20.0;
uniform float glow : hint_range(0.0, 2.0) = 0.9;
uniform float density : hint_range(0.2, 2.0) = 1.0;
uniform float age_fade : hint_range(0.0, 1.0) = 0.35;

varying flat float v_age;
varying flat float v_seed;
varying vec2 v_uv;

// --- Smooth value noise -----------------------------------------------------
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
	// Billboard: quad center in world space, offset onto camera plane.
	vec3 origin = (MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec3 cam_right = vec3(VIEW_MATRIX[0][0], VIEW_MATRIX[1][0], VIEW_MATRIX[2][0]);
	vec3 cam_up = vec3(VIEW_MATRIX[0][1], VIEW_MATRIX[1][1], VIEW_MATRIX[2][1]);

	// Instance scale from the packed Vec4 (see wind_tunnel_flow.gdshader).
	vec3 scale = MODEL_MATRIX[0].xyz;
	float r = length(scale) * puff_scale;
	vec3 offset = (cam_right * VERTEX.x + cam_up * VERTEX.y) * 2.0 * r;

	POSITION = PROJECTION_MATRIX * (VIEW_MATRIX * vec4(origin + offset, 1.0));
	v_age = COLOR.r;
	v_seed = COLOR.g * 256.0 + COLOR.b;
	v_uv = VERTEX.xy;
}

void fragment() {
	// Soft radial puff profile (broad so neighbours overlap into volume).
	float d = length(v_uv) * 2.0;
	float prof = 1.0 - smoothstep(1.0 - puff_soft, 1.0, d);
	prof *= prof * prof;

	// World position of the pixel.
	vec3 wp = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;

	// Streaming shimmer bands riding downstream at the live wind, so the air
	// keeps visibly rushing even in near-uniform flow.
	float s = -wp.x;
	float w1 = sin(s * 1.1 - TIME * min(flow_speed * 1.2, 45.0) + v_seed * 41.7);
	float w2 = sin(s * 2.6 - TIME * min(flow_speed * 2.4, 70.0) + v_seed * 93.3);
	float travel = 0.5 + 0.5 * (0.65 * w1 + 0.35 * w2);
	travel = pow(travel, 1.4);

	// Wispy cloud mottling — world noise seeded per puff, advected.
	vec3 bp = wp * noise_scale + v_seed * 37.0;
	bp.z += TIME * noise_speed;
	float n0 = vnoise(bp);
	float n1 = vnoise(bp * 2.1 + 1.7);
	float n2 = vnoise(bp * 4.3 + 3.1);
	float mottle = 0.5 + 0.5 * (n0 * 0.5 + n1 * 0.3 + n2 * 0.2);

	float brightness = mix(travel, mottle, 0.55);

	// Age: fresh at the rake (opaque, smaller-cored) -> dissipating downstream.
	float age = clamp(v_age, 0.0, 1.0);
	float alpha = prof * core_alpha * brightness * density;
	alpha *= 1.0 - age_fade * age * age;

	vec3 col = mix(base_color, aged_color, age * 0.6);

	ALPHA = clamp(alpha, 0.0, 1.0);
	ALBEDO = col;
	EMISSION = col * (0.25 + alpha * 0.7) * glow;
}
