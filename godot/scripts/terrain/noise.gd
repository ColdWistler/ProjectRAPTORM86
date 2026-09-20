class_name TerrainNoise
extends RefCounted
## Fractal Perlin (FBM) height noise — a Godot port of the `Noise.cs` from
## Sebastian Lague's *Procedural Landmass Generation* series (E03+).
##
## `generate_noise_map(w, h, seed, scale, octaves, persistence, lacunarity,
## offset)` becomes a `get_height(x, z)` query able to evaluate the heightmap
## at any world position, which is what makes chunk edges tile seamlessly
## (neighbours sample the same global noise at identical coordinates).

var seed_value := 1337:
	set(v):
		seed_value = v
		_noise.seed = v
var scale := 120.0:
	set(v):
		scale = maxf(v, 0.0001)
		_noise.frequency = 1.0 / scale
var octaves := 4:
	set(v):
		octaves = maxi(v, 1)
		_noise.fractal_octaves = octaves
var persistence := 0.5:
	set(v):
		persistence = v
		_noise.fractal_gain = persistence
var lacunarity := 2.0:
	set(v):
		lacunarity = maxf(v, 1.0)
		_noise.fractal_lacunarity = lacunarity
var offset := Vector2.ZERO

var _noise := FastNoiseLite.new()

func _init(
	p_seed: int = 1337,
	p_scale: float = 120.0,
	p_octaves: int = 4,
	p_persistence: float = 0.5,
	p_lacunarity: float = 2.0,
) -> void:
	_noise.noise_type = FastNoiseLite.TYPE_PERLIN
	_noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	_noise.fractal_weighted_strength = 0.2
	seed_value = p_seed
	scale = p_scale
	octaves = p_octaves
	persistence = p_persistence
	lacunarity = p_lacunarity

## Normalised FBM Perlin height in `[0, 1]` at world (x, z) in metres.
## Matches the `InverseLerp(minNoise, maxNoise, value)` output of E03.
func get_height(x: float, z: float) -> float:
	var n := _noise.get_noise_2d(x + offset.x, z + offset.y)
	return (clampf(n, -1.0, 1.0) + 1.0) * 0.5