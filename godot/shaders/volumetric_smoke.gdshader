// Volumetric wind-tunnel AIR TRAILS.
//
// Each MultiMesh instance is a unit QuadMesh oriented in the vertex shader to
// form a short tube segment:
//   * the quad is stretched along the streak's local tangent (local +Y of the
//     instance basis, projected onto the camera plane) so consecutive
//     instances overlap into one continuous filament,
//   * a soft cross-section profile (`v_object.y`) rounds the tube into a true
//     volume rather than a flat card or a dot,
//   * quad ends are faded so segments blend seamlessly end-to-end,
//   * world-space value noise is STRETCHED along the tangent and advected
//     with TIME, so the churn flows as fibrous, streaky air -- not random
//     mottling,
//   * age (0 newest at the rake -> 1 oldest downstream) widens and dissipates
//     the trail downstream.
//
// Per-instance custom data: x = age, y = per-link seed.

shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_disabled, unshaded;

// --- Tuning -----------------------------------------------------------------
uniform vec3 base_color : source_color = vec3(0.88, 0.97, 1.0);
uniform vec3 aged_color : source_color = vec3(0.62, 0.78, 0.93);
uniform float core_alpha : hint_range(0.0, 1.0) = 0.85;    // filament core opacity
uniform float edge_soft : hint_range(0.0, 1.0) = 0.45;     // cross-section profile softness (thin thread)
uniform float end_fade : hint_range(0.0, 1.0) = 0.08;      // segment end fade (0 hard -> 1 long blend)
uniform float trail_stretch : hint_range(0.5, 4.0) = 4.0;  // elongation along the flow (x radius)
uniform float fiber_mix : hint_range(0.0, 1.0) = 0.35;     // how much thread detail vs clean streaming waves
uniform float noise_scale : hint_range(0.01, 1.0) = 0.14;  // thread frequency in world space
uniform float noise_speed : hint_range(0.0, 2.0) = 0.6;    // thread drift speed
uniform float flow_speed : hint_range(0.0, 200.0) = 20.0;  // live wind speed the streaming waves ride on
uniform float flow_streak : hint_range(0.0, 1.0) = 0.85;   // stretch the threads along the trail
uniform float age_fade : hint_range(0.0, 1.0) = 0.35;      // dissipation downstream
uniform float glow : hint_range(0.0, 1.0) = 0.6;           // emissive glow of the thread

varying vec3 v_object;
varying flat vec3 v_tangent;
varying flat float v_age;
varying flat float v_seed;

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
	// Instance origin and its scaled local offset (basis scaled by the tube
	// radius, so VERTEX.x/y in [-0.5, 0.5] map to [±r] along trail / across).
	vec3 origin = (MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec3 off = (MODEL_MATRIX * vec4(VERTEX, 0.0)).xyz;

	// Camera right/up in world space (columns 0/1 of the view matrix).
	vec3 cam_right = vec3(VIEW_MATRIX[0][0], VIEW_MATRIX[1][0], VIEW_MATRIX[2][0]);
	vec3 cam_up = vec3(VIEW_MATRIX[0][1], VIEW_MATRIX[1][1], VIEW_MATRIX[2][1]);

	// Local +Y of the instance = streak tangent (normalized; basis is scaled).
	vec3 tangent = normalize(MODEL_MATRIX[1].xyz);

	// Project the tangent onto the camera plane to get the screen trail axis.
	float tx = dot(tangent, cam_right);
	float ty = dot(tangent, cam_up);
	float tl = length(vec2(tx, ty));
	vec2 axis = tl > 1e-4 ? vec2(tx, ty) / tl : vec2(1.0, 0.0);
	vec2 perp = vec2(-axis.y, axis.x);
	vec3 trail_dir = cam_right * axis.x + cam_up * axis.y;
	vec3 cross_dir = cam_right * perp.x + cam_up * perp.y;

	// Stretch along the trail, keep the across-axis at the tube radius.
	vec3 world_pos = origin + trail_dir * off.x * trail_stretch + cross_dir * off.y;
	POSITION = PROJECTION_MATRIX * (VIEW_MATRIX * vec4(world_pos, 1.0));

	v_object = VERTEX;
	v_tangent = tangent;
	v_age = INSTANCE_CUSTOM.x;
	v_seed = INSTANCE_CUSTOM.y;
}

void fragment() {
	// Round tube cross-section profile from the across-axis local coord.
	float cross = v_object.y * 2.0;             // -1 (left edge) .. 1 (right edge)
	float prof = 1.0 - smoothstep(1.0 - edge_soft, 1.0, abs(cross));

	// Blend the quad ends so consecutive segments form one seamless trail.
	float t = v_object.x * 2.0;                 // -1 (upstream end) .. 1 (downstream end)
	float ef = 0.05 + end_fade * 0.5;
	float ends = smoothstep(-1.0, -1.0 + ef * 2.0, t) * (1.0 - smoothstep(1.0 - ef * 2.0, 1.0, t));

	// World position of the pixel (VERTEX is view-space here).
	vec3 wp = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;

	// STREAMING MOTION. Three octaves of luminance waves anchored in the LAB
	// frame that travel downstream at speeds around the live wind speed. In
	// steady uniform flow the advected field is time-invariant, so without
	// this the picture would be frozen: the bands visualise the wind rushing
	// through the chamber. The sine phases are offset per segment via the
	// seed so adjacent filaments shimmer independently.
	float s = dot(wp, v_tangent);
	float om = flow_speed * 1.5;
	float w1 = sin(s * 0.8 - TIME * min(om * 0.9, 40.0) + v_seed * 41.7);
	float w2 = sin(s * 1.9 - TIME * min(om * 1.7, 55.0) + v_seed * 93.3);
	float w3 = sin(s * 3.6 - TIME * min(om * 3.0, 70.0) + v_seed * 27.3);
	float travel = 0.5 + 0.5 * (0.45 * w1 + 0.35 * w2 + 0.2 * w3);
	travel = pow(travel, 1.7);              // contrastier eddies, clearer motion

	// Thread detail: streaky fibers stretched along the trail that slowly
	// drift and evolve downstream. Deliberately NOT locked to `flow_speed` --
	// a comoving pattern would look frozen.
	vec3 base = wp * noise_scale;
	vec3 np = base + v_tangent * (dot(base, v_tangent) * flow_streak);
	np += v_seed * 37.0;
	np += v_tangent * (TIME * noise_speed);
	float n0 = vnoise(np);
	float n1 = vnoise(np * 2.0 + 1.7);
	float fiber = 0.5 + 0.5 * (n0 * 0.65 + n1 * 0.35);

	float density = mix(travel, fiber, fiber_mix);

	float alpha = prof * ends * (core_alpha * density);

	// Age-based dissipation downstream.
	float age = clamp(v_age, 0.0, 1.0);
	float fade = 1.0 - age_fade * age * age;
	alpha *= fade;

	vec3 col = mix(base_color, aged_color, age * 0.6);

	ALPHA = clamp(alpha, 0.0, 1.0);
	ALBEDO = col;
	EMISSION = col * alpha * glow;
}