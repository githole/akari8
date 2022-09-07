#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <optional>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "vec3.h"

#define PI 3.141592654f
namespace utility
{
    constexpr float kLarge = 1e+32f;
//using Vec3 = std::array<float, 3>;

//using Color = std::array<float, 3>;
using Color = Float3;

template<typename T>
T clampValue(const T& x, const T& a, const T& b)
{
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

template<typename V>
float dot(const V& a, const V& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<typename V>
V normalizeUnsafe(const V& vec)
{
    const auto length2{ dot(vec, vec) };
    const auto invLength{ 1 / sqrt(length2) };

    return {
        vec[0] * invLength,
        vec[1] * invLength,
        vec[2] * invLength,
    };
}

template<typename V>
float lengthSquared(const V& a, const V& b)
{
    return
        (a[0] - b[0]) * (a[0] - b[0]) +
        (a[1] - b[1]) * (a[1] - b[1]) +
        (a[2] - b[2]) * (a[2] - b[2]);
}

struct Image
{
    std::vector<float> body;
    int width{};
    int height{};

    Image() = default;

    Image(int w, int h) : width(w), height(h)
    {
        body.resize(w * h * 3);
    }

    bool isValid() const
    {
        return !body.empty();
    }

    size_t clampedIndex(int x, int y) const
    {
        if (x <= 0)
            x = 0;
        if (x >= width)
            x = width - 1;
        if (y <= 0)
            y = 0;
        if (y >= height)
            y = height - 1;
        return ((size_t)x + (size_t)y * width) * 3;
    }

    Color load(int x, int y) const
    {
        const auto index{ clampedIndex(x, y) };
        return {
            body[index + 0],
            body[index + 1],
            body[index + 2],
        };
    }
    
    Color load2(int x, int y) const
    {
        if (x < 0 || y < 0 || x >= width || y >= height)
        {
            return {};
        }

        return load(x, y);
    }

    void store(int x, int y, const Color& color)
    {
        const auto index{ clampedIndex(x, y) };
        body[index + 0] = color[0];
        body[index + 1] = color[1];
        body[index + 2] = color[2];
    }

    void accum(int x, int y, const Color& color)
    {
        const auto index{ clampedIndex(x, y) };
        body[index + 0] += color[0];
        body[index + 1] += color[1];
        body[index + 2] += color[2];
    }


    void accum2(int x, int y, const Color& color)
    {
        if (x < 0 || y < 0 || x >= width || y >= height)
        {
            return;
        }

        const auto index{ clampedIndex(x, y) };
        body[index + 0] += color[0];
        body[index + 1] += color[1];
        body[index + 2] += color[2];
    }

    void accumBilinear(float u, float v, const Color& color)
    {
        const float fu = u * width - 0.5f;
        const float fv = v * height - 0.5f;

        const int iu = (int)fu;
        const float wu = fu - iu;
        const int iv = (int)fv;
        const float wv = fv - iv;

        accum2(iu, iv, (1 - wv) * (1 - wu) * color);
        accum2(iu + 1, iv, (1 - wv) * wu * color);
        accum2(iu, iv + 1, wv * (1 - wu) * color);
        accum2(iu + 1, iv + 1, wv * wu * color);
    }

    template<bool edgeZero = true>
    Color loadBilinear(float u, float v) const
    {
        const float fu = u * width - 0.5f;
        const float fv = v * height - 0.5f;

        const int iu = (int)fu;
        const float wu = fu - iu;
        const int iv = (int)fv;
        const float wv = fv - iv;

        if constexpr (edgeZero == 0)
        {
            const auto v00 = load2(iu, iv);
            const auto v10 = load2(iu + 1, iv);
            const auto v01 = load2(iu, iv + 1);
            const auto v11 = load2(iu + 1, iv + 1);

            return
                (1 - wv) * ((1 - wu) * v00 + wu * v10) +
                wv * ((1 - wu) * v01 + wu * v11);
        }
        else
        {

            const auto v00 = load(iu, iv);
            const auto v10 = load(iu + 1, iv);
            const auto v01 = load(iu, iv + 1);
            const auto v11 = load(iu + 1, iv + 1);

            return
                (1 - wv) * ((1 - wu) * v00 + wu * v10) +
                wv * ((1 - wu) * v01 + wu * v11);
        }
    }
};

inline Image loadHDRImage(const char* filename)
{
    Image image;
    int component{};
    constexpr int reqComponent{ 3 };
    const float* data{ stbi_loadf(filename, &image.width, &image.height, &component, reqComponent) };
    if (!data)
    {
        return {};
    }
    const size_t elementCount{ (size_t)reqComponent * image.width * image.height };
    image.body.reserve(elementCount);
    image.body.insert(image.body.begin(), data, data + elementCount);
    stbi_image_free((void*)data);

    std::cout << "Loaded: " << filename << std::endl;
    return image;
}

// true if succeeded.
inline bool writeHDRImage(const char* filename, const Image& image)
{
    return stbi_write_hdr(filename, image.width, image.height, 3, image.body.data());
}

inline bool dumpHDRImage(const char* filename, const Image& image)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp)
    {
        return false;
    }

    uint32_t w = image.width;
    fwrite(&w, sizeof(uint32_t), 1, fp);
    uint32_t h = image.height;
    fwrite(&h, sizeof(uint32_t), 1, fp);

    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.height; ++ix)
        {
            const auto col = image.load(ix, iy);
            fwrite(col.v, sizeof(float), 3, fp);
        }
    }
}

inline Image readDumpedHDRImage(const char* filename)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        return {};
    }

    uint32_t w, h;
    fread(&w, sizeof(uint32_t), 1, fp);
    fread(&h, sizeof(uint32_t), 1, fp);
    Image image(w, h);

    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.width; ++ix)
        {
            Color col;
            fread(col.v, sizeof(float), 3, fp);
            image.store(ix, iy, col);
        }
    }
    return image;
}

inline bool checkResolutionEquality(const Image& a, const Image& b)
{
    return a.width == b.width && a.height == b.height;
}


namespace random {

    uint32_t rotr(uint32_t x, int shift) {
        return (x >> shift) | (x << (32 - shift));
    }

    uint64_t rotr(uint64_t x, int shift) {
        return (x >> shift) | (x << (64 - shift));
    }

    struct splitmix64 {
        uint64_t x;

        splitmix64(uint64_t a = 0) : x(a) {}

        uint64_t next() {
            uint64_t z = (x += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        }
    };

    // PCG(64/32)
    // http://www.pcg-random.org/download.html
    // initial_inc from official library
    struct PCG_64_32 {
        uint64_t state;
        uint64_t inc;

        PCG_64_32(uint64_t initial_state = 0x853c49e6748fea9bULL,
            uint64_t initial_inc = 0xda3e39cb94b95bdbULL)
            : state(initial_state), inc(initial_inc) {}

        void set_seed(uint64_t seed) {
            splitmix64 s(seed);
            state = s.next();
        }

        using return_type = uint32_t;
        return_type next() {
            auto oldstate = state;
            state = oldstate * 6364136223846793005ULL + (inc | 1);
            uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;

            return rotr(xorshifted, rot);
        }

        // [0, 1)
        float next01() {
            return (float)(((double)next()) /
                ((double)std::numeric_limits<uint32_t>::max() + 1));
        }

        float next(float minV, float maxV)
        {
            return next01() * (maxV - minV) + minV;
        }
    };

} // namespace random



template <typename Vec3>
inline void createOrthoNormalBasis(const Vec3& normal, Vec3* tangent, Vec3* binormal) {
    if (abs(normal[0]) > abs(normal[1]))
    {
        (*tangent) = cross(Vec3(0, 1, 0), normal);
        (*tangent) = normalize(*tangent);
    }
    else
    {
        (*tangent) = cross(Vec3(1, 0, 0), normal);
        (*tangent) = normalize(*tangent);
    }
    (*binormal) = cross(normal, *tangent);
    (*binormal) = normalize(*binormal);
}

struct Material
{
    enum Type
    {
        Glass,
        Diffuse
    } type;
};

struct Matrix3x3
{
    float m[9];
};


struct Primitive
{
    virtual const Material* getMaterial() const = 0;

    virtual const Matrix3x3& getMatrix() const = 0;
};

struct Hitpoint
{
    float distance{ kLarge };
    Float3 position; // world space
    Float3 normal; // world space

    float uv[2]{};
    bool enableTex{ false };

    const Primitive* primitive{ nullptr };
};

struct Ray
{
    Float3 org;
    Float3 dir;
};

struct Sphere {
    float radius;
    Float3 position;
    Color emission;
    Color color;


    Sphere(const double radius, const Float3& position, const Color& emission, const Color& color) :
        radius(radius), position(position), emission(emission), color(color) {}

    // 入力のrayに対する交差点までの距離を返す。交差しなかったら0を返す。
    // rayとの交差判定を行う。交差したらtrue,さもなくばfalseを返す。
    bool intersect(const Ray& ray, Hitpoint& hitpoint) const {
        const Float3 p_o = position - ray.org;
        const double b = dot(p_o, ray.dir);
        const double D4 = b * b - dot(p_o, p_o) + radius * radius;

        if (D4 < 0.0)
            return false;

        const double sqrt_D4 = sqrt(D4);
        const double t1 = b - sqrt_D4, t2 = b + sqrt_D4;

        constexpr float kEPS = 1e-6f;

        if (t1 < kEPS && t2 < kEPS)
            return false;

        if (t1 > kEPS) {
            hitpoint.distance = t1;
        }
        else {
            hitpoint.distance = t2;
        }

        hitpoint.position = ray.org + hitpoint.distance * ray.dir;
        hitpoint.normal = normalize(hitpoint.position - position);
        return true;
    }
};

Matrix3x3 createIdentity()
{
    Matrix3x3 mat;
    mat.m[0] = 1; mat.m[1] = 0; mat.m[2] = 0;
    mat.m[3] = 0; mat.m[4] = 1; mat.m[5] = 0;
    mat.m[6] = 0; mat.m[7] = 0; mat.m[8] = 1;
    return mat;
}

Matrix3x3 createRotX(float t)
{
    Matrix3x3 mat;
    const auto c = cos(t);
    const auto s = sin(t);
    mat.m[0] = 1; mat.m[1] = 0; mat.m[2] = 0;
    mat.m[3] = 0; mat.m[4] = c; mat.m[5] = -s;
    mat.m[6] = 0; mat.m[7] = s; mat.m[8] = c;
    return mat;
}

Matrix3x3 createRotY(float t)
{
    Matrix3x3 mat;
    const auto c = cos(t);
    const auto s = sin(t);
    mat.m[0] = c; mat.m[1] = 0; mat.m[2] = s;
    mat.m[3] = 0; mat.m[4] = 1; mat.m[5] = 0;
    mat.m[6] =-s; mat.m[7] = 0; mat.m[8] = c;
    return mat;
}

Matrix3x3 createRotZ(float t)
{
    Matrix3x3 mat;
    const auto c = cos(t);
    const auto s = sin(t);
    mat.m[0] = c; mat.m[1] =-s; mat.m[2] = 0;
    mat.m[3] = s; mat.m[4] = c; mat.m[5] = 0;
    mat.m[6] = 0; mat.m[7] = 0; mat.m[8] = 1;
    return mat;
}

Float3 applyMatrix(const Matrix3x3& mat, const Float3& v)
{
    return {
        mat.m[0] * v[0] + mat.m[1] * v[1] + mat.m[2] * v[2],
        mat.m[3] * v[0] + mat.m[4] * v[1] + mat.m[5] * v[2],
        mat.m[6] * v[0] + mat.m[7] * v[1] + mat.m[8] * v[2],
    };
}

Matrix3x3 mult(const Matrix3x3& a, const Matrix3x3& b)
{
    const Vec3 a0 = applyMatrix(a, { b.m[0], b.m[3], b.m[6] });
    const Vec3 a1 = applyMatrix(a, { b.m[1], b.m[4], b.m[7] });
    const Vec3 a2 = applyMatrix(a, { b.m[2], b.m[5], b.m[8] });

    Matrix3x3 mat;
    mat.m[0] = a0[0]; mat.m[1] = a1[0]; mat.m[2] = a2[0];
    mat.m[3] = a0[1]; mat.m[4] = a1[1]; mat.m[5] = a2[1];
    mat.m[6] = a0[2]; mat.m[7] = a1[2]; mat.m[8] = a2[2];
    return mat;
}

Matrix3x3 inv(const Matrix3x3& a)
{
    const float a11 = a.m[0];
    const float a12 = a.m[1];
    const float a13 = a.m[2];
    const float a21 = a.m[3];
    const float a22 = a.m[4];
    const float a23 = a.m[5];
    const float a31 = a.m[6];
    const float a32 = a.m[7];
    const float a33 = a.m[8];

    const float D = 1.0f /
        (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32);

    Matrix3x3 mat;

    mat.m[0] = a22 * a33 - a23 * a32; mat.m[1] = -(a12 * a33 - a13 * a32); mat.m[2] = a12 * a23 - a13 * a22;
    mat.m[3] = -(a21 * a33 - a23 * a31); mat.m[4] = a11 * a33 - a13 * a31; mat.m[5] = -(a11 * a23 - a13 * a21);
    mat.m[6] = a21 * a32 - a22 * a31; mat.m[7] = -(a11 * a32 - a12 * a31); mat.m[8] = a11 * a22 - a12 * a21;

    for (int i = 0; i < 9; ++i)
        mat.m[i] *= D;

    return mat;
}


struct AABB
{
    Float3 bounds[2];
    Float3 center;

    AABB(const Float3& vmin, const Float3& vmax)
    {
        bounds[0] = vmin;
        bounds[1] = vmax;
        center = (bounds[0] + bounds[1]) * 0.5f;
    }

    AABB()
    {
        for (int i = 0; i < 3; ++i)
        {
            bounds[0][i] = kLarge;
            bounds[1][i] = -kLarge;
        }
    }

    void merge(const AABB& aabb)
    {
        for (int i = 0; i < 3; ++i) 
        {
            bounds[0][i] = std::min(bounds[0][i], aabb.bounds[0][i]);
            bounds[1][i] = std::max(bounds[1][i], aabb.bounds[1][i]);
        }
    }

    std::optional<Hitpoint> intersect(const Ray& ray) const {
        auto tmphp = intersect_(ray);

        if (tmphp && tmphp->distance >= 0)
        {
            return tmphp;
        }

        return {};
    }

    std::optional<Hitpoint> intersect_(const Ray& ray) const {
        float tmin, tmax, tymin, tymax, tzmin, tzmax;


        Float3 invdir(1.0f/ ray.dir[0], 1.0f / ray.dir[1], 1.0f / ray.dir[2]);
        int sign[3];
        sign[0] = (invdir[0] < 0);
        sign[1] = (invdir[1] < 0);
        sign[2] = (invdir[2] < 0);

        tmin = (bounds[sign[0]][0] - ray.org[0]) * invdir[0];
        tmax = (bounds[1 - sign[0]][0] - ray.org[0]) * invdir[0];
        tymin = (bounds[sign[1]][1] - ray.org[1]) * invdir[1];
        tymax = (bounds[1 - sign[1]][1] - ray.org[1]) * invdir[1];

        if ((tmin > tymax) || (tymin > tmax))
            return {};

        int axis = 0;

        if (tymin > tmin)
        {
            axis = 1;
            tmin = tymin;
        }
        if (tymax < tmax)
            tmax = tymax;

        tzmin = (bounds[sign[2]][2] - ray.org[2]) * invdir[2];
        tzmax = (bounds[1 - sign[2]][2] - ray.org[2]) * invdir[2];

        if ((tmin > tzmax) || (tzmin > tmax))
            return {};

        if (tzmin > tmin)
        {
            axis = 2;
            tmin = tzmin;
        }
        if (tzmax < tmax)
            tmax = tzmax;

        Hitpoint hitpoint;
        hitpoint.distance = tmin > 0 ? tmin : tmax;
        hitpoint.position = ray.org + hitpoint.distance * ray.dir;

        Float3 normal(0.0f, 0.0f, 0.0f);
        normal[axis] = 1.0f;
        if (center[axis] > hitpoint.position[axis])
        {
            normal *= -1.0f;
        }
        hitpoint.normal = normal;

        return hitpoint;
    }
};


struct Box : public Primitive
{
    Float3 center;
    Float3 span;
    Matrix3x3 rot;
    Matrix3x3 invrot;
    AABB aabb;
    const Material* material;

    bool enableSpecialEffect = false;

    Box(const Float3& vcenter, const Float3& vspan, const Matrix3x3 vrot, const Material* vmaterial = nullptr)
    {
        center = vcenter;
        span = vspan;
        rot = vrot;
        invrot = inv(rot);
        aabb = createAABB_();
        material = vmaterial;
    }

    const Material* getMaterial() const override
    {
        return material;
    }

    const Matrix3x3& getMatrix() const override
    {
        return rot;
    }

    bool inside(const Float3& pt) const
    {
        const auto p = pt - center;
        const auto localP = applyMatrix(invrot, p);


        if (-span[0] <= localP[0] &&
            -span[1] <= localP[1] &&
            -span[2] <= localP[2] &&
            localP[0] <= span[0] &&
            localP[1] <= span[1] &&
            localP[2] <= span[2])
        {
            return true;
        }
        return false;
    }

    std::optional<Hitpoint> intersect(const Ray& raytmp) const {
        Ray ray;
        ray.org = raytmp.org - center;
        ray.org = applyMatrix(invrot, ray.org);
        ray.dir = applyMatrix(invrot, raytmp.dir);

        AABB aabb(-span, span);

        const auto tmphp = aabb.intersect(ray);
        if (!tmphp)
        {
            return {};
        }

        Hitpoint hitpoint;

        if (enableSpecialEffect)
        {
            if (length2(tmphp->normal - Float3(0, -1, 0)) < 1e-5f)
            {
                // 底面 (u, v)
                const auto pos = ray.org + tmphp->distance * ray.dir;
                hitpoint.uv[0] = (pos[0] + span[0]) / (2.0f * span[0]);
                hitpoint.uv[1] = (pos[2] + span[2]) / (2.0f * span[2]);
                hitpoint.enableTex = true;
            }
        }

        hitpoint.distance = tmphp->distance;
        hitpoint.position = raytmp.org + tmphp->distance * raytmp.dir;
        hitpoint.normal = applyMatrix(rot, tmphp->normal);
        hitpoint.primitive = this;
        return hitpoint;
    }

    AABB createAABB_() const
    {
        Float3 vmin(kLarge, kLarge, kLarge);
        Float3 vmax = -vmin;

        for (int i = 0; i < 8; ++i)
        {
            Float3 v;

            v[0] = span[0] * ((i & 1) ? 1 : -1);
            v[1] = span[1] * ((i & 2) ? 1 : -1);
            v[2] = span[2] * ((i & 4) ? 1 : -1);

            v = applyMatrix(rot, v) + center;

            for (int c = 0; c < 3; ++c)
            {
                vmin[c] = std::min(vmin[c], v[c]);
                vmax[c] = std::max(vmax[c], v[c]);
            }
        }

        return AABB(vmin, vmax);
    }
};

Float3 sample_uniform_sphere_surface(float u, float v) {
    const float tz = u * 2 - 1;
    const float phi = v * PI * 2;
    const float k = sqrt(1.0 - tz * tz);
    const float tx = k * cos(phi);
    const float ty = k * sin(phi);
    return Float3(tx, ty, tz);
}

// https://twitter.com/matt_zucker/status/980528965243748352
Float3 sample_cos_weighted(float u, float v, const Float3& normal) {
    return normalize(normal + sample_uniform_sphere_surface(u, v));
}


float easeOutBack(float x)
{
    const float c1 = 1.70158;
    const float c3 = c1 + 1;

    return 1 + c3 * pow(x - 1, 3) + c1 * pow(x - 1, 2);
}

float easeOutQuad(float x)
{
    return 1 - (1 - x) * (1 - x);
}

float easeInExpo(float x)
{
return x == 0 ? 0 : pow(2, 10 * x - 10);
}

float easeOutQuint(float x)
{
    return 1 - pow(1 - x, 5);
}

float easeOutExpo(float x)
{
    return x == 1 ? 1 : 1 - pow(2, -10 * x);
}

}
