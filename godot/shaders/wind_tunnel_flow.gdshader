// GPU wind-tunnel particle advection shader.
//
// Advects every particle through the VLM-grade flow field entirely on the GPU.
// The flow field is uploaded each step from Rust as a Texture3D.  Particle
// motion, aging, recycling, and packing all happen in this process stage — no
// per-particle CPU work, no MultiMesh buffer transfers.
//
// The engine integrates VELOCITY into the transform each frame, so we set
// VELOCITY to a flow-field sample and let the engine move the particle.
//
// Per-particle output data is packed into COLOR for the spatial render pass:
//   R = age_norm (0 fresh -> 1 dissipating)
//   G,B = 24-bit per-particle seed (decorrelates noise between puffs)

shader_type particles;

// --- Flow field (uploaded each step from Rust) -----------------------------
uniform sampler3D flow_tex;
uniform vec3 flow_origin;
uniform vec3 flow_cell;
uniform vec3 flow_dims;
uniform float vmax = 20.0;

// --- Rake geometry (spawn / recycle plane) ---------------------------------
uniform vec3 rake_pos = vec3(9.0, 0.0, 0.0);
uniform vec3 rake_extents = vec3(0.4, 5.0, 5.0);

// --- Particle lifetime (seconds) -------------------------------------------
uniform float particle_life = 2.0;

// --- Puff size (world metres; the render shader reads the X basis column) ---
uniform float puff_radius = 0.45;

// Per-particle PRNG (LCG seeded from INDEX/RANDOM_SEED): takes a uint state
// and returns a float in [0, 1), advancing the state.
float rnd(inout uint s) {
	s = s * 747796405u + 2891336453u;
	uint r = (s >> uint(13)) ^ s;
	return float(r & 0xffffu) / 65535.0;
}

// World-space -> flow-volume UV (0..1 across the grid).
vec3 world_to_uv(vec3 wp) {
	vec3 cell_idx = (wp - flow_origin) / (flow_cell * max(flow_dims - 1.0, 1.0));
	return clamp(cell_idx, 0.0, 1.0);
}

// Trilinear velocity sample at a world position, unscaled to m/s.
vec3 sample_flow(vec3 wp) {
	vec4 v = texture(flow_tex, world_to_uv(wp));
	return v.rgb * vmax;
}

// Random rake position within the spawn box, from an advancing PRNG state.
vec3 random_rake_pos(inout uint s) {
	vec3 o = vec3(rnd(s) - 0.5, rnd(s) - 0.5, rnd(s) - 0.5) * 2.0;
	return rake_pos + o * rake_extents;
}

void process() {
	// Stable per-particle identity from the particle index + random seed.
	uint s = uint(INDEX) * 2654435761u + RANDOM_SEED;
	float sval = fract(float(INDEX) * 0.0001 + float(RANDOM_SEED % 97u) * 0.01);

	if (RESTART) {
		// Staggered age so the stream fills the volume instead of appearing
		// all at once, then a random start on the rake.
		float life_frac = rnd(s);
		TRANSFORM[3].xyz = random_rake_pos(s);
		TRANSFORM[0] = vec4(puff_radius, 0.0, 0.0, 0.0);
		TRANSFORM[1] = vec4(0.0, puff_radius, 0.0, 0.0);
		TRANSFORM[2] = vec4(0.0, 0.0, 1.0, 0.0);
		VELOCITY = vec3(0.0);

		// Pack age + seed.
		COLOR.r = life_frac;
		COLOR.g = fract(sval * 256.0);
		COLOR.b = fract(sval * 65536.0);
		COLOR.a = 1.0;
	} else {
		// Advance age using the fraction of lifetime already used (R channel).
		float age = COLOR.r + DELTA / particle_life;

		// Recycle expired particles back to the rake.
		if (age >= 1.0) {
			COLOR.r = 0.0;
			VELOCITY = vec3(0.0);
			TRANSFORM[3].xyz = random_rake_pos(s);
		} else {
			// Advect: sample the flow and let the engine integrate VELOCITY.
			vec3 pos = TRANSFORM[3].xyz;
			VELOCITY = sample_flow(pos);

			// Recycle out-of-bounds particles.
			if (pos.x < flow_origin.x + flow_cell.x * 0.5
					|| pos.y < flow_origin.y - flow_cell.y
					|| pos.y > flow_origin.y + flow_cell.y * (flow_dims.y + 1.0)
					|| abs(pos.z) > flow_cell.z * (flow_dims.z * 0.5 + 1.0)) {
				TRANSFORM[3].xyz = random_rake_pos(s);
				VELOCITY = vec3(0.0);
				age = 0.0;
			}

			COLOR.r = age;
		}
	}
}
