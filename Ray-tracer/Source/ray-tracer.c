#ifndef _RAYTRACER_
#define _RAYTRACER_
#define _CRT_SECURE_NO_WARNINGS


#include <math.h>
#include <stdlib.h> 
#include <time.h>
#include <stdio.h> 
#include <float.h>


// Recursion depth
#define	MAX_RAY_DEPTH	50
#define	PI				3.14159265358979323846 
#define true			1
#define false			0


// TO DO (Vasilen): implement a RNG that returns a double/float precision value in the range 0 <= random_value < 1 , there is overhead on the division after rand() , which can be avoided

typedef	int bool_t;
typedef struct { float x, y, z; } vec3_t;
typedef struct { vec3_t origin, dir; } ray_t;
typedef struct { float fuzz, ref_index; vec3_t albedo; bool_t reflective; bool_t refractive; } material_t;
typedef struct { vec3_t center;  float radius; material_t material; } sphere_t;
typedef struct { float t; vec3_t p, normal; material_t material; } hit_record_t;
typedef struct { vec3_t lower_left_corner, horizontal, vertical, origin, u, v, w; float lens_radius; } camera_t;

static inline vec3_t Scale(const vec3_t vec, float a) { return (vec3_t) { a * vec.x, a * vec.y, a * vec.z }; }
static inline vec3_t Negate(const vec3_t a) { return (vec3_t){ -a.x, -a.y, -a.z }; }
static inline vec3_t Multiply(const vec3_t a, const vec3_t b) { return (vec3_t) { a.x * b.x, a.y * b.y, a.z * b.z }; }
static inline vec3_t DivideScalar(const vec3_t a, float t) { float k = 1.0f / t; return (vec3_t) { a.x * k, a.y * k, a.z * k }; }
static inline float Length(const vec3_t a) { return (float)sqrt((a.x * a.x) + (a.y * a.y) + (a.z * a.z));  }
static inline float SquaredLength(const vec3_t a) { return (a.x * a.x) + (a.y * a.y) + (a.z * a.z); }
static inline vec3_t Normalize(const vec3_t vec) { float len = 1.0f / Length(vec); return (vec3_t) { vec.x * len, vec.y * len, vec.z * len }; }
static inline vec3_t Subtract(const vec3_t a, const vec3_t b) { return (vec3_t){ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline vec3_t Add(const vec3_t a, const vec3_t b) { return (vec3_t) { a.x + b.x, a.y + b.y,  a.z + b.z }; }
static inline vec3_t AddScalar(const vec3_t a, float f) { return (vec3_t) { a.x + f, a.y + f, a.z + f }; }
static inline float Dot( const vec3_t a, const vec3_t b) { return ( a.x * b.x ) + ( a.y * b.y ) + ( a.z * b.z ); }
static inline vec3_t Cross(const vec3_t a, const vec3_t b){ vec3_t result; result.x = (a.y * b.z - a.z * b.y); result.y = (a.z * b.x - a.x * b.z); result.z = (a.x * b.y - a.y * b.x); return result; }
static inline vec3_t Lerp(const vec3_t v0, const vec3_t v1, float t) { return (vec3_t) { (1.0f - t) * v0.x + t * v1.x, (1.0f - t) * v0.y + t * v1.y, (1.0f - t) * v0.z + t * v1.z }; }
static inline float LerpFloat(float v0, float v1, float t) { return (1.0f - t) * v0 + t * v1; }

// TO DO (Vasilen) : try a xorshift random number generating ?
inline float RandomNormValue(void)
{
	return ( rand() / (RAND_MAX + 1.0f) );
}

vec3_t RandomInUnitSphere(void)
{
	vec3_t p = { 0.0f, 0.0f, 0.0f };
	do
	{
		vec3_t random = { RandomNormValue(), RandomNormValue(), RandomNormValue() };
		vec3_t max = { 1.0f, 1.0f, 1.0f};
		p = Scale(Subtract(random, max), 2.0f);
	} while (SquaredLength(p) >= 1.0f);
	return p;
}

vec3_t RandomInUnitDisk(void)
{
	vec3_t p = { 0.0f, 0.0f, 0.0f };
	do
	{
		vec3_t random = { RandomNormValue(), RandomNormValue(), 0.0f };
		vec3_t max = { 1.0f, 1.0f, 0.0f };
		p = Subtract(Scale(random, 2.0f), max);
	} while (Dot(p, p) >= 1.0f);
	return p;
}

ray_t ConstructRay(float s, float t, const camera_t *camera)
{
	vec3_t rd = Scale(RandomInUnitDisk(), camera->lens_radius);
	vec3_t offset = Add(Scale(camera->u, rd.x), Scale(camera->v, rd.y));
	vec3_t sh = Scale(camera->horizontal, s);
	vec3_t tv = Scale(camera->vertical, t);

	vec3_t dir = Subtract(Subtract(Add(Add(camera->lower_left_corner, sh), tv), camera->origin), offset);
	ray_t r = { Add(camera->origin, offset), dir };
	return r;
}

camera_t ConstructCamera(vec3_t eye, vec3_t lookat, vec3_t up, float vfov, float aspect, float aperture, float focus_dist)
{
	camera_t camera;
	
	camera.lens_radius = aperture / 2.0f;
	float theta = vfov * (float)PI / 180.0f;
	float half_height = tanf(theta / 2.0f);
	float half_width = aspect * half_height;
	camera.origin = eye;
	camera.w = Subtract(eye, lookat);
	camera.w = Normalize(camera.w);
	camera.u = Normalize(Cross(up, camera.w));
	camera.v = Cross(camera.w, camera.u);
	
	// lower_left_corner = origin - half_width*u - half_height*v - w;
	vec3_t us = Scale(camera.u, half_width * focus_dist);
	vec3_t vs = Scale(camera.v, half_height * focus_dist);
	camera.lower_left_corner = Subtract( Subtract( Subtract(camera.origin, us), vs ), Scale(camera.w, focus_dist));
	camera.horizontal = Scale(camera.u, 2.0f * half_width * focus_dist);
	camera.vertical = Scale(camera.v, 2.0f * half_height * focus_dist);
	
	return camera;
}


vec3_t PointAtParameter(const ray_t *ray, const float t)
{
	return Add( Scale(ray->dir, t), ray->origin);
}

float Schlick(float cosine, float ref_index)
{
	float r0 = (1.0f - ref_index) / (1.0f + ref_index);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf( (1.0f - cosine), 5.0f );
}

bool_t HitSphere(const sphere_t *sphere, float t_min, float t_max, const ray_t *r, hit_record_t *rec)
{
	vec3_t oc = Subtract(r->origin, sphere->center);
	float a = Dot(r->dir, r->dir);
	float b = Dot(oc, r->dir);
	float c = Dot(oc, oc) - sphere->radius * sphere->radius;
	float discriminant = b * b - a * c;

	if (discriminant > 0.0f)
	{
		float x1 = (-b - (float)sqrt(b * b - a * c)) / a;
		if (x1 < t_max && x1 > t_min)
		{
			rec->t = x1;
			rec->p = PointAtParameter(r, rec->t);
			rec->normal = DivideScalar(Subtract(rec->p, sphere->center), sphere->radius);
			return true;
		}
		x1 = (-b + (float)sqrt(b * b - a * c)) / a;
		if (x1 < t_max && x1 > t_min)
		{
			rec->t = x1;
			rec->p = PointAtParameter(r, rec->t);
			rec->normal = DivideScalar(Subtract(rec->p, sphere->center), sphere->radius);
			return true;
		}
	}
	return false;
}
vec3_t Reflect(const vec3_t* v, const vec3_t* n)
{
	// v - 2 * dot(v, n) * n
	return Subtract( *v, Scale( Scale( *n, Dot(*v, *n) ), 2.0f ) );
}

bool_t Refract(const vec3_t *v, const vec3_t *n, float ni_over_nt, vec3_t *refracted)
{
	vec3_t uv = Normalize(*v);
	float dt = Dot(uv, *n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
	if (discriminant > 0.0f)
	{
		// refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant)
		*refracted = Subtract(Scale(Subtract(uv, Scale(*n, dt)), ni_over_nt), Scale(*n, (float)sqrt(discriminant)) );
		return true;
	}
	else
	{
		return false;
	}
}

bool_t RefractionScatter(const ray_t *r_in, const hit_record_t *rec, vec3_t *attenuation, ray_t *scattered)
{
	vec3_t outward_normal;
	vec3_t reflected = Reflect(&r_in->dir, &rec->normal);
	vec3_t refracted = {0.0f, 0.0f, 0.0f};
	float ni_over_nt;
	*attenuation = (vec3_t){ 1.0f, 1.0f, 1.0f };
	float reflect_prob = 0.0f;
	float cosine = 0.0f;
	if (Dot(r_in->dir, rec->normal) > 0.0f)
	{
		outward_normal = Negate(rec->normal);
		ni_over_nt = rec->material.ref_index;
		cosine = rec->material.ref_index * Dot(r_in->dir, rec->normal) / Length(r_in->dir);
	}
	else
	{
		outward_normal = rec->normal;
		ni_over_nt = 1.0f / rec->material.ref_index;
		cosine = -( Dot(r_in->dir, rec->normal) ) / Length(r_in->dir);
	}

	if (Refract(&r_in->dir, &outward_normal, ni_over_nt, &refracted))
	{
		reflect_prob = Schlick(cosine, rec->material.ref_index);
	}
	else
	{
		*scattered = (ray_t){ rec->p, reflected };
		reflect_prob = 1.0f;
	}

	if (RandomNormValue() < reflect_prob)
	{
		*scattered = (ray_t){ rec->p, reflected };
	}
	else
	{
		*scattered = (ray_t){ rec->p, refracted };
	}
	return true;
}

bool_t RayTraceWorld(const ray_t* r, float t_min, float t_max, sphere_t* spheres[], unsigned int size_spheres, hit_record_t* rec)
{
	hit_record_t temp_rec;
	bool_t hit_anything = false;
	float closest_so_far = t_max;
	for (unsigned int i = 0; i < size_spheres; ++i)
	{
		if (HitSphere(spheres[i], t_min, closest_so_far, r, &temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			temp_rec.material = spheres[i]->material;
			// Clamping so there is no scattering below the surface
			if (spheres[i]->material.fuzz > 1.0f)
			{ 
				spheres[i]->material.fuzz = 1.0f;
				temp_rec.material.fuzz = 1.0f;
			}
			*rec = temp_rec;
		}
	}
	return hit_anything;
}

vec3_t Render(const ray_t* r, sphere_t* spheres[], unsigned int size_spheres, int depth)
{
	hit_record_t rec;
	if (RayTraceWorld(r, 0.001f, FLT_MAX, spheres, size_spheres, &rec) && depth < MAX_RAY_DEPTH)
	{
		// It's a reflective material ( metal )
		if (rec.material.reflective)
		{
			vec3_t temp_norm = Normalize(r->dir);
			vec3_t reflected = Reflect(&temp_norm, &rec.normal);
			ray_t scattered = { rec.p, Add(reflected, Scale(RandomInUnitSphere(), rec.material.fuzz) ) };
			if (Dot(scattered.dir, rec.normal) > 0)
			{
				return Multiply(Render(&scattered, spheres, size_spheres, depth + 1), rec.material.albedo);
			}
			else
			{
				vec3_t nothing = { 0.0f, 0.0f, 0.0f };
				return nothing;
			}
			
		}
		else if (rec.material.refractive)
		{
			ray_t scattered;
			vec3_t attenuation;
			if(RefractionScatter(r, &rec, &attenuation, &scattered))
			{
				return Multiply(Render(&scattered, spheres, size_spheres, depth + 1), attenuation);
			}
			else
			{
				return (vec3_t){ 0.0f, 0.0f, 0.0f };
			}
	}

		// It's a diffuse material 
		else
		{
			vec3_t target = Add(Add(rec.p, rec.normal), RandomInUnitSphere());
			ray_t material_ray = { rec.p , Subtract(target, rec.p) };
			return Multiply(Render(&material_ray, spheres, size_spheres, depth + 1), rec.material.albedo);
		}
	}
	else
	{
		vec3_t unit_direction = Normalize(r->dir);
		vec3_t v0 = { 1.0f, 1.0f, 1.0f };
		vec3_t v1 = { 0.5f, 0.7f, 1.0f };
		float f = 0.5f * (unit_direction.y + 1.0f);
		return Lerp(v0, v1, f);
	}
}

sphere_t* CreateSphere(vec3_t center, float radius, bool_t reflective, bool_t refractive, float refrection_index, float reflection_fuzz, vec3_t albedo)
{
	sphere_t* sphere = (sphere_t*)malloc(sizeof(sphere_t));
	sphere->center = center;
	sphere->radius = radius;
	sphere->material.reflective = reflective;
	sphere->material.refractive = refractive;
	sphere->material.ref_index = refrection_index;
	sphere->material.fuzz = reflection_fuzz;
	sphere->material.albedo = albedo;
	return sphere;
}

int main(int argc, char **argv)
{
	srand( (unsigned int)time(NULL) );   // should only be called once

	// Sphere array init
	sphere_t* hitables[4];
	
	hitables[0] = CreateSphere( (vec3_t) { -1.5f, 0.0f, -1.0f }, 0.5f, false, true, 1.5f, 0.0f, (vec3_t){ 0.0f, 0.0f, 0.0f } );
	hitables[1] = CreateSphere((vec3_t) { -1.5f, 0.0f, -1.0f }, -0.45f, false, true, 1.5f, 0.0f, (vec3_t) { 0.0f, 0.0f, 0.0f } );
	hitables[2] = CreateSphere((vec3_t) { 0.0f, 0.0f, -1.0f }, 0.5f, false, false, 0.0f, 0.0f, (vec3_t) { 0.1f, 0.2f, 0.5f } );
	hitables[3] = CreateSphere((vec3_t) { 1.5f, 0.0f, -1.0f }, 0.5f, true, false, 0.0f, 0.0f, (vec3_t) { 0.8f, 0.6f, 0.2f } );
	// "Ground" sphere
	hitables[1] = CreateSphere( (vec3_t){ 0.0f, -100.5f, -1.0f }, 100.0f, false, false, 0.0f, 0.0f, (vec3_t){ 0.8f, 0.8f, 0.0f } );
		
	vec3_t eye = { -8.0f, 2.0f, 8.0f };
	vec3_t lookat = { 0.0f, 0.0f, 0.0f };
	vec3_t up = { 0.0f, 1.0f, 0.0f };
	float dist_to_focus = Length(Subtract(eye, lookat));
	float aperture = 0.1f;

	int nx = 1200;
	int ny = 800;
	int ns = 10;

	FILE *image = fopen("test.ppm", "wb"); /* b - binary mode */
	(void)fprintf(image, "P6\n%d %d\n255\n", nx, ny);

	printf("PPM\n255\n%d, %d\n", nx, ny);
	printf("Rendering...\n");

	camera_t camera = ConstructCamera(eye, lookat, up, 40.0f, (float)(nx) / (float)(ny), aperture, dist_to_focus);

	for (int j = ny - 1; j >= 0; --j)
	{
		for (int i = 0; i < nx; ++i)
		{
			vec3_t color = { 0.0f, 0.0f, 0.0f };
			for (int s = 0; s < ns; ++s)
			{
				float u = (float)(i + RandomNormValue()) / (float)(nx);
				float v = (float)(j + RandomNormValue()) / (float)(ny);
				ray_t r = ConstructRay(u, v, &camera);
				color = Add( color, Render( &r, hitables, sizeof(hitables) / sizeof(sphere_t*), 0) );
			}

			color = DivideScalar(color, (float)ns);
			color = (vec3_t){ (float)sqrt(color.x) , (float)sqrt(color.y), (float)sqrt(color.z) };
			
			static unsigned char rgb[3];
			rgb[0] = (char)(255.99f * color.x);
			rgb[1] = (char)(255.99f * color.y);
			rgb[2] = (char)(255.99f * color.z);

			(void)fwrite(rgb, 1, 3, image);

			//ofs << ir << " " << ig << " " << ib << "\n";
		}
	}
	(void)fclose(image);

	// Clean up
	for (int z = 0; z < sizeof(hitables) / sizeof(sphere_t*); ++z)
	{
		free(hitables[z]);
	}
	//system("PAUSE");
}

#endif