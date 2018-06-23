#include <cstdlib> 
#include <time.h>
#include <cstdio> 
#include <fstream> 
#include <iostream>
#include <float.h>

// Recursion depth
#define MAX_RAY_DEPTH 5

// TO DO (Vasilen): implement a RNG that returns a double/float precision value in the range 0 <= random_value < 1 , the rand() is probably shit
// TO DO (Vasilen): implement a memory allocator, probably it's gonna be fun and it'll look way better than malloc ( and perform way better )

// TO DO (Vasilen)(Fixed): For some fucking reason the recursion algorith is not working ( maybe ). It renders whatever is first in the array , and then the second on top
typedef struct { float x, y, z; } vec3_t;
typedef struct { float reflection; vec3_t albedo; } material_t;
typedef struct { vec3_t origin, dir; } ray_t;
typedef struct { vec3_t center;  float radius; material_t material; } sphere_t;
typedef struct { float t; vec3_t p, normal; material_t material; } hit_record_t;
typedef struct { vec3_t lower_left_corner, horizontal , vertical , origin; } camera_t;

static inline vec3_t Scale(const vec3_t &vec, float a) { vec3_t result = vec;  result.x *= a, result.y *= a, result.z *= a; return result; }
static inline vec3_t Multiply(const vec3_t &a, const vec3_t &b) { vec3_t result = a; result.x *= b.x, result.y *= b.y, result.z *= b.z; return result; }
static inline vec3_t DivideScalar(const vec3_t &a, float t) { float k = 1.0f / t; vec3_t result = a;  result.x *= k, result.y *= k, result.z *= k; return result; }
static inline float Length(const vec3_t &a) { return sqrt((a.x * a.x) + (a.y * a.y) + (a.z * a.z));  }
static inline float SquaredLength(const vec3_t &a) { return (a.x * a.x) + (a.y * a.y) + (a.z * a.z); }
static inline vec3_t Normalize(const vec3_t &a) { vec3_t result = a;  float len = 1.0f / Length(result); result.x *= len; result.y *= len; result.z *= len; return result; }
static inline vec3_t Subtract(const vec3_t &a, const vec3_t &b) { vec3_t result = { a.x - b.x, a.y - b.y, a.z - b.z }; return result; }
static inline vec3_t Add(const vec3_t &a, const vec3_t &b) { vec3_t result = a; result.x += b.x;  result.y += b.y;  result.z += b.z; return result; }
static inline vec3_t AddScalar(const vec3_t &a, float f) { vec3_t result = a; result.x += f;  result.y += f;  result.z += f; return result; }
static inline float Dot(const vec3_t &a, const vec3_t &b) { return ( a.x * b.x ) + ( a.y * b.y ) + ( a.z * b.z ); }
static inline vec3_t Lerp(const vec3_t& v0, const vec3_t& v1, float t) { vec3_t result = { (1.0f - t) * v0.x + t * v1.x, (1.0f - t) * v0.y + t * v1.y, (1.0f - t) * v0.z + t * v1.z }; return result; }
static inline float LerpFloat(float v0, float v1, float t) { return (1.0f - t) * v0 + t * v1; }

inline void ResetVec3(vec3_t &a) { a = { 5, 6, 7 }; }
inline void PrintVec3(vec3_t &a) { printf("%f, %f, %f\n", a.x, a.y, a.z); }
inline void PrintResetVec3(vec3_t& a) { PrintVec3(a);  ResetVec3(a); }

ray_t ConstructRay(float u, float v, const camera_t &camera)
{
	vec3_t dir = Add(camera.lower_left_corner, Add(Scale(camera.horizontal, u), Scale(camera.vertical, v)));
	ray_t r = { camera.origin, dir };
	return r;
}

float RandomNormValue()
{
	//return ( rand() % (1 + 1- 0) + 0 );
	return ( rand() / (RAND_MAX + 1.0f) );
}

vec3_t PointAtParameter(const ray_t &ray, const float t)
{
	return Add( Scale(ray.dir, t), ray.origin);	
}

bool HitSphere(const sphere_t &sphere, float t_min, float t_max, const ray_t& r, hit_record_t &rec)
{
	vec3_t oc = Subtract(r.origin, sphere.center);
	float a = Dot(r.dir, r.dir);
	float b = Dot(oc, r.dir);
	float c = Dot(oc, oc) - sphere.radius * sphere.radius;
	float discriminant = b * b - a * c;

	if (discriminant > 0.0f)
	{
		float x1 = (-b - sqrt(b * b - a * c)) / a;
		if (x1 < t_max && x1 > t_min)
		{
			rec.t = x1;
			rec.p = PointAtParameter(r, rec.t);
			rec.normal = DivideScalar(Subtract(rec.p, sphere.center), sphere.radius);
			return true;
		}
		x1 = (-b + sqrt(b * b - a * c)) / a;
		if (x1 < t_max && x1 > t_min)
		{
			rec.t = x1;
			rec.p = PointAtParameter(r, rec.t);
			rec.normal = DivideScalar(Subtract(rec.p, sphere.center), sphere.radius);
			return true;
		}
	}
	return false;
}

vec3_t RandomInUnitSphere()
{
	vec3_t p = { };
	do
	{
		vec3_t random = { RandomNormValue(), RandomNormValue(), RandomNormValue() };
		vec3_t max = { 1.0f, 1.0f, 1.0f};
		p = Scale( Subtract(random, max), 2.0f );
	} while (SquaredLength(p) >= 1.0f);
	return p;
}

bool Scatter(const ray_t& r_in, const hit_record_t& rec, vec3_t& attenuation, ray_t& scattered)
{
	// Metal or any reflective type of material ?
	if (rec.material.reflection > 0.0f)
	{
		return 0;
	}
	//It's a diffuse material
	else
	{
		vec3_t target = Add(Add(rec.p, rec.normal), RandomInUnitSphere());
		scattered = { rec.p , Subtract(target, rec.p) };
		attenuation = rec.material.albedo;
		return true;
	}
	return false;
}

vec3_t Reflect(const vec3_t& v, const vec3_t& n)
{
	// v - 2 * dot(v, n) * n
	return Subtract( v, Scale( Scale( n, Dot(v, n) ), 2.0f ) );
}

bool RayTraceWorld(const ray_t& r, float t_min, float t_max, sphere_t* spheres[], size_t size_spheres, hit_record_t &rec)
{
	hit_record_t temp_rec = { };
	bool hit_anything = false;
	float closest_so_far = t_max;
	for (unsigned int i = 0; i < size_spheres; ++i)
	{
		if (HitSphere(*spheres[i], t_min, closest_so_far, r, temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			temp_rec.material.albedo = spheres[i]->material.albedo;
			temp_rec.material.reflection = spheres[i]->material.reflection;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

vec3_t Render(const ray_t& r, sphere_t* spheres[], size_t size_spheres, int depth)
{
	hit_record_t rec = { };
	if (RayTraceWorld(r, 0.001f, FLT_MAX, spheres, size_spheres, rec) && depth < MAX_RAY_DEPTH)
	{
		// It's a reflective material ( metal )
		if (rec.material.reflection > 0.0f)
		{
			vec3_t reflected = Reflect(Normalize(r.dir), rec.normal);
			ray_t scattered = { rec.p, reflected };
			if (Dot(scattered.dir, rec.normal) > 0)
			{
				return Multiply(Render(scattered, spheres, size_spheres, depth + 1), rec.material.albedo);
			}
		}
		// It's a diffuse material 
		else
		{
			vec3_t target = Add(Add(rec.p, rec.normal), RandomInUnitSphere());
			ray_t material_ray = { rec.p , Subtract(target, rec.p) };
			return Multiply(Render(material_ray, spheres, size_spheres, depth + 1), rec.material.albedo);
		}
	}
	else
	{
		vec3_t unit_direction = Normalize(r.dir);
		vec3_t v0 = { 1.0f, 1.0f, 1.0f };
		vec3_t v1 = { 0.5f, 0.7f, 1.0f };
		float f = 0.5f * (unit_direction.y + 1.0f);
		return Lerp(v0, v1, f);
	}
}

void CreateSphere(sphere_t* sphere, vec3_t center, float radius, float reflection, vec3_t albedo)
{
	sphere = (sphere_t*)malloc(sizeof(sphere_t));
	sphere->center = center;
	sphere->radius = radius;
	sphere->material.reflection = reflection;
	sphere->material.albedo = albedo;
}

int main(int argc, char **argv)
{
	int nx = 200;
	int ny = 100;
	int ns = 100;
	srand( (unsigned int)time(NULL) );   // should only be called once

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	std::ofstream ofs("./render.ppm", std::ios::out | std::ios::binary);
	ofs << "P3\n" << nx << " " << ny << "\n255\n";
	camera_t camera = { };
	camera.lower_left_corner = { -2.0f, -1.0f, -1.0f };
	camera.horizontal = { 4.0f, 0.0f, 0.0f };
	camera.vertical = { 0.0f, 2.0f, 0.0f };
	camera.origin = { 0.0f, 0.0f, 0.0f };

	// Sphere array init
	sphere_t *hitables[4];
	hitables[0] = (sphere_t*)malloc(sizeof(sphere_t));
	hitables[0]->center = { 0.0f, 0.0f, -1.0f };
	hitables[0]->radius = 0.5f;
	hitables[0]->material.reflection = 0.0f;
	hitables[0]->material.albedo = { 0.8f, 0.3f, 0.3f};


	hitables[1] = (sphere_t*)malloc(sizeof(sphere_t));
	hitables[1]->center = { 0.0f, -100.5f, -1.0f };
	hitables[1]->radius = 100.0f;
	hitables[1]->material.reflection = 0.0f;
	hitables[1]->material.albedo = { 0.8f, 0.8f, 0.0f };

	hitables[2] = (sphere_t*)malloc(sizeof(sphere_t));
	hitables[2]->center = { 1.0f, 0.0f, -1.0f };
	hitables[2]->radius = 0.5f;
	hitables[2]->material.reflection = 1.0f;
	hitables[2]->material.albedo = { 0.8f, 0.6f, 0.2f };

	hitables[3] = (sphere_t*)malloc(sizeof(sphere_t));
	hitables[3]->center = { -1.0f, 0.0f, -1.0f };
	hitables[3]->radius = 0.5f;
	hitables[3]->material.reflection = 1.0f;
	hitables[3]->material.albedo = { 0.8f, 0.8f, 0.8f };

	//std::cout << sizeof(hitables) / sizeof(sphere_t*) << std::endl;
	std::cout << "Rendering..."<< std::endl;
	for (int j = ny - 1; j >= 0; --j)
	{
		for (int i = 0; i < nx; ++i)
		{
			vec3_t color = { 0.0f, 0.0f, 0.0f };
			for (int s = 0; s < ns; ++s)
			{
				float u = float(i + RandomNormValue()) / float(nx);
				float v = float(j + RandomNormValue()) / float(ny);
				ray_t r = ConstructRay(u, v, camera);
				vec3_t p = { PointAtParameter(r, 2.0f) };
				
				color = Add( color, Render( r, hitables, sizeof(hitables) / sizeof(sphere_t*), 0) );
			}

			color = DivideScalar(color, (float)ns);
			color = { sqrt(color.x) , sqrt(color.y), sqrt(color.z) };
			int ir = (int)(255.99f * color.x);
			int ig = (int)(255.99f * color.y);
			int ib = (int)(255.99f * color.z);

			ofs << ir << " " << ig << " " << ib << "\n";
		}
	}
	ofs.close();
	
	// Clean up
	for (int z = 0; z < sizeof(hitables) / sizeof(sphere_t*); ++z)
	{
		free(hitables[z]);
	}
	//system("PAUSE");
}

