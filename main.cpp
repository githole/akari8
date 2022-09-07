#include "utility.h"

#include <optional>

constexpr bool kRough = false;

utility::Image jacobbi(
    const utility::Image& D,
    float kThreshold,
    int kMaxIteration,
    float dx,
    float scale)
{
    utility::Image tmp0(D.width, D.height);
    utility::Image tmp1(D.width, D.height);

    auto get = [](const utility::Image& phi, int x, int y)
    {
        if (x < 0 || phi.width <= x ||
            y < 0 || phi.height <= y)
        {
            return 0.0f;
        }
        return phi.load(x, y)[0];
    };

    utility::Image* current_phi = &tmp0;
    utility::Image* next_phi = &tmp0;

//    float MaxPhi = 1.0e-10f;

    std::vector<float> maxPhi(D.height, 1.0e-10f);
    std::vector<float> maxError(D.height);

    for (int iteartion = 0; iteartion < kMaxIteration; ++iteartion)
    {
        std::fill(maxError.begin(), maxError.end(), 0.0f);

#pragma omp parallel for schedule(dynamic, 1)
        for (int iy = 0; iy < D.height; ++iy)
        {
            for (int ix = 0; ix < D.width; ++ix)
            {
                const auto d = get(D, ix, iy);
                const auto p0 = get(*current_phi, ix + 1, iy);
                const auto p1 = get(*current_phi, ix - 1, iy);
                const auto p2 = get(*current_phi, ix, iy + 1);
                const auto p3 = get(*current_phi, ix, iy - 1);
                const float prev = current_phi->load(ix, iy)[0];
                const float next = 0.25f * (dx * dx * d + p0 + p1 + p2 + p3);

                next_phi->store(ix, iy, {
                    next, 0, 0
                    });

                if (maxPhi[iy] < fabs(next))
                {
                    maxPhi[iy] = next;
                }

                const float tmp = fabs(next - prev);
                if (maxError[iy] < tmp)
                {
                    maxError[iy] = tmp;
                }
            }
        }

        float currentMaxPhi = 1e-10f;
        for (int iy = 0; iy < D.height; ++iy)
        {
            if (currentMaxPhi < maxPhi[iy])
            {
                currentMaxPhi = maxPhi[iy];
            }
        }

        float currentMaxError = 0.0f;
        for (int iy = 0; iy < D.height; ++iy)
        {
            const float err = maxError[iy] / currentMaxPhi;
            if (currentMaxError < err)
            {
                currentMaxError = err;
            }
        }

        std::swap(current_phi, next_phi);

        if (!(iteartion % 1000)) printf("%05d  %e %e\n", iteartion, currentMaxPhi, currentMaxError);

        if (currentMaxError < kThreshold)
        {
            printf("%05d  %e %e\n", iteartion, currentMaxPhi, currentMaxError);
            break;
        }
    }

    for (int iy = 0; iy < D.height; ++iy)
    {
        for (int ix = 0; ix < D.width; ++ix)
        {
            const auto d = scale * current_phi->load(ix, iy)[0];
            current_phi->store(ix, iy, { d, d, d });
        }
    }
    return *current_phi;
}


utility::Image gauss_sidel(
    const utility::Image& D,
    float kThreshold,
    int kMaxIteration, 
    float dx,
    float scale)
{
    utility::Image tmp0(D.width, D.height);
    utility::Image tmp1(D.width, D.height);

    auto get = [](const utility::Image& phi, int x, int y)
    {
        if (x < 0 || phi.width <= x ||
            y < 0 || phi.height <= y)
        {
            return 0.0f;
        }
        return phi.load(x, y)[0];
    };

    utility::Image* current_phi = &tmp0;
    utility::Image* next_phi = &tmp0;

    float MaxPhi = 1.0e-10f;
    for (int iteartion = 0; iteartion < kMaxIteration; ++iteartion)
    {
        float MaxError = 0.0f;

        for (int iy = 0; iy < D.height; ++iy)
        {
            for (int ix = 0; ix < D.width; ++ix)
            {
                const auto d = get(D, ix, iy);
                const auto p0 = get(*current_phi, ix + 1, iy);
                const auto p1 = get(*current_phi, ix - 1, iy);
                const auto p2 = get(*current_phi, ix, iy + 1);
                const auto p3 = get(*current_phi, ix, iy - 1);
                const float prev = current_phi->load(ix, iy)[0];
                const float next = 0.25f * (dx * dx * d + p0 + p1 + p2 + p3);

                next_phi->store(ix, iy, {
                    next, 0, 0
                });

                if (MaxPhi < fabs(next))
                {
                    MaxPhi = next;
                }
                const float CurrentError = (fabs(next - prev)) / MaxPhi;

                if (MaxError < CurrentError)
                {
                    MaxError = CurrentError;
                }
            }
        }
        std::swap(current_phi, next_phi);

        if (!(iteartion % 1000)) printf("%05d  %e %e\n", iteartion, MaxPhi, MaxError);

        if (MaxError < kThreshold)
        {
            printf("%05d  %e %e\n", iteartion, MaxPhi, MaxError);
            break;
        }
    }

    for (int iy = 0; iy < D.height; ++iy)
    {
        for (int ix = 0; ix < D.width; ++ix)
        {
            const auto d = scale * current_phi->load(ix, iy)[0];
            current_phi->store(ix, iy, { d, d, d });
        }
    }
    return *current_phi;
}

float average(const utility::Image& image)
{
    double sum = 0;
    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.width; ++ix)
        {
            sum += image.load(ix, iy)[0];
        }
    }
    return sum / (image.width * image.height);
}

utility::Image luminanceToRedBlue(const utility::Image& image)
{
    utility::Image result(image.width, image.height);

    for (int iy = 0; iy < result.height; ++iy)
    {
        for (int ix = 0; ix < result.width; ++ix)
        {
            const float d = image.load(ix, iy)[0];
            if (d > 0)
            {
                result.store(ix, iy, { d, 0, 0 });
            }
            else
            {
                result.store(ix, iy, { 0, 0, -d });
            }
        }
    }
    return result;
}

struct Point
{
    float x, y; // [m]
    float u, v;
};

void advect(const utility::Image& phi, float PhysicalSize, float dx, std::vector<Point>& points, float dt)
{
    utility::Image vectorField(phi.width, phi.height);

    for (int iy = 0; iy < phi.height; ++iy)
    {
        for (int ix = 0; ix < phi.width; ++ix)
        {
            float vec[2];
            vec[0] = (phi.load2(ix + 1, iy)[0] - phi.load2(ix - 1, iy)[0]) / (2 * dx);
            vec[1] = (phi.load2(ix, iy + 1)[0] - phi.load2(ix, iy - 1)[0]) / (2 * dx);
            vectorField.store(ix, iy, { vec[0], vec[1], 0.0f });
        }
    }

    // advect

    for (auto& p : points)
    {
        const auto vec = vectorField.loadBilinear(p.x / PhysicalSize, p.y / PhysicalSize);
        p.x += dt * vec[0];
        p.y += dt * vec[1];
    }
}

void savePoints(const char* filename, const std::vector<Point>& points, int imageWidth, float PhysicalSize)
{
    utility::Image image(imageWidth, imageWidth);

    for (auto& p : points)
    {
        float x = p.x / PhysicalSize;
        float y = p.y / PhysicalSize;
        image.accumBilinear(x, y, utility::Vec3(1.0f, 1.0f, 1.0f));
    }

    utility::writeHDRImage(filename, image);
}

utility::Image createMapping(const std::vector<Point>& points, int imageWidth, float PhysicalSize)
{
    utility::Image image(imageWidth, imageWidth);
    for (auto& p : points)
    {
        float x = p.x / PhysicalSize;
        float y = p.y / PhysicalSize;
        image.accumBilinear(p.x, p.y, utility::Vec3(p.u, p.v, 1.0f));
    }

    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.width; ++ix)
        {
            auto n = image.load(ix, iy);
            if (n[2] > 0)
            {
                n[0] /= n[2];
                n[1] /= n[2];
                n[2] = 0;
            }
            image.store(ix, iy, n);
        }
    }
    return image;
}

utility::Image applyMapping(const utility::Image& mapping, float PhysicalSize)
{
    utility::Image image(mapping.width, mapping.height);

    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.width; ++ix)
        {
            const auto uv = mapping.load(ix, iy);
            image.accumBilinear(uv[0], uv[1], utility::Float3(1.0f, 1.0f, 1.0f));
        }
    }
    return image;
}

utility::Image computeNormalXY(const std::vector<Point>& points, const utility::Image& mapping, const utility::Image& hImage, float H, float eta, float PhysicalSize)
{
    utility::Image normalXY(hImage.width, hImage.height);
#if 1
    for (int iy = 0; iy < normalXY.height; ++iy)
    {
        for (int ix = 0; ix < normalXY.width; ++ix)
        {
            const float x = (ix + 0.5f) / normalXY.width;
            const float y = (iy + 0.5f) / normalXY.height;
            const float u = mapping.load(ix, iy)[0];
            const float v = mapping.load(ix, iy)[1];

            const auto h = hImage.loadBilinear(u / PhysicalSize, v / PhysicalSize)[0];
            const float a = (u - x) * (u - x) + (v - y) * (v - y);
            const float b = H - h;
            const float k = eta * sqrt(a + b * b) - b;

            utility::Color normal;
            normal[0] = (u - x) / k;
            normal[1] = (v - y) / k;
            normal[2] = 1;

            normalXY.accumBilinear(x, y, normal);
#if 0

            ////
            const float z = 3.0f;

            const bool into = false; // レイがオブジェクトから出るのか、入るのか

            utility::Float3 org(x, y, z);
            utility::Float3 dir(0, 0, -1);

            normal = normalize(normal);
            const auto orienting_normal = -normal;

            // Snellの法則
            const float nc = 1.0; // 真空の屈折率
            const float nt = 1.5f; // オブジェクトの屈折率
            const float nnt = into ? nc / nt : nt / nc;
            const float ddn = dot(dir, normal);
            const float cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

            // 屈折の方向
            const auto refraction_dir =
                normalize(dir * nnt - normal * (ddn * nnt + sqrt(cos2t)));

            float t = -org[2] / refraction_dir[2];

            const auto hitP = org + t * refraction_dir;

            const float u2 = hitP[0] / PhysicalSize;
            const float v2 = hitP[1] / PhysicalSize;


            printf("[%f, %f] - [%f, %f]\n", u, v, u2, v2);
#endif
        }
    }
#endif

#if 0
    for (auto& p : points)
    {
        const auto h = hImage.loadBilinaer(p.u / PhysicalSize, p.v / PhysicalSize)[0];
        const float a = (p.u - p.x) * (p.u - p.x) + (p.v - p.y) * (p.v - p.y);
        const float b = H - h;
        const float k = eta * sqrt(a + b * b) - b;

        utility::Color normal;
        normal[0] = (p.u - p.x) / k;
        normal[1] = (p.v - p.y) / k;
        normal[2] = 1;

        const float x = p.u / PhysicalSize;
        const float y = p.v / PhysicalSize;
        normalXY.accumBilinear(x, y, normal);
    }
#endif

    for (int iy = 0; iy < normalXY.height; ++iy)
    {
        for (int ix = 0; ix < normalXY.width; ++ix)
        {
            auto n = normalXY.load(ix, iy);
            if (n[2] > 0)
            {
                n[0] /= n[2];
                n[1] /= n[2];
                n[2] = 0;
            }
            normalXY.store(ix, iy, n);
        }
    }

    return normalXY;
}

utility::Image diversity(const utility::Image& input, float dx, float scale)
{
    utility::Image divImage(input.width, input.height);

    for (int iy = 0; iy < input.height; ++iy)
    {
        for (int ix = 0; ix < input.width; ++ix)
        {
            float dv[2];
            dv[0] = (input.load2(ix + 1, iy)[0] - input.load2(ix - 1, iy)[0]) / (2 * dx);
            dv[1] = (input.load2(ix, iy + 1)[1] - input.load2(ix, iy - 1)[1]) / (2 * dx);

            const float d = scale * (dv[0] + dv[1]);
            divImage.store(ix, iy, { d, d, d });
        }
    }

    return divImage;
}

utility::Image gradient(const utility::Image& input, float dx)
{
    utility::Image gradImage(input.width, input.height);

    for (int iy = 0; iy < input.height; ++iy)
    {
        for (int ix = 0; ix < input.width; ++ix)
        {
            float dv[2];
            dv[0] = (input.load2(ix + 1, iy)[0] - input.load2(ix - 1, iy)[0]) / (2 * dx);
            dv[1] = (input.load2(ix, iy + 1)[1] - input.load2(ix, iy - 1)[1]) / (2 * dx);

            gradImage.store(ix, iy, { dv[0], dv[1], 0.0f});
        }
    }

    return gradImage;
}

utility::Image generatePhotonImage(const std::vector<Point>& points, const utility::Image& normalImage, float scale, float normalScale)
{
    constexpr float kIor = 1.5f;
    constexpr float PhysicalSizePlane = 1.0f;

#if 0
    // validate
    for (auto& pt : points)
    {
        const float x = pt.x;
        const float y = pt.y;
        const float z = 3.0f;

        const bool into = false; // レイがオブジェクトから出るのか、入るのか

        utility::Float3 org(x, y, z);
        utility::Float3 dir(0, 0, -1);

        auto normal = normalImage.loadBilinaer(x, y);
        normal[2] = -1;
        normal = normalize(normal);

        // Snellの法則
        const float nc = 1.0; // 真空の屈折率
        const float nt = kIor; // オブジェクトの屈折率
        const float nnt = into ? nc / nt : nt / nc;
        const float ddn = dot(dir, normal);
        const float cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

        /*
        if (cos2t < 0.0) { // 全反射
            incoming_radiance = radiance(reflection_ray, rnd, depth + 1);
            weight = now_object.color / russian_roulette_probability;
            break;
        }
        */
        // 屈折の方向
        const auto refraction_dir =
            normalize(dir * nnt + normal * (ddn * nnt + sqrt(cos2t)));


        float t = -refraction_dir[2] / org[2];

        const auto hitP = org + t * refraction_dir;

        const float u = hitP[0] / PhysicalSizePlane;
        const float v = hitP[1] / PhysicalSizePlane;


        printf("[%f, %f] - [%f, %f]\n", pt.u, pt.v, u, v);
    }
#endif



    // ray tracing
    constexpr float PhysicalSize = 1.0f;
#if 0
    auto hImage = utility::loadHDRImage("./hImage2.hdr");
    for (int iy = 0; iy < hImage.height; ++iy)
    {
        for (int ix = 0; ix < hImage.width; ++ix)
        {
            auto h = hImage.load(ix, iy);
            if (h[0] > 0)
            {
                hImage.store(ix, iy, { h[0], h[0], h[0] });
            }
            else if (h[2] > 0)
            {
                hImage.store(ix, iy, { -h[2], -h[2], -h[2] });
            }
        }
    }

    auto normalImage = gradient(hImage, PhysicalSize / hImage.width);
#endif
    constexpr int accumImageSize = 128;
    utility::Image accumImage(accumImageSize, accumImageSize);

    // photon
    constexpr int PhotonDiv = 1024;
    for (int iy = 0; iy < PhotonDiv; ++iy)
    {
        for (int ix = 0; ix < PhotonDiv; ++ix)
        {
            const float x = (ix + 0.5f) / PhotonDiv;
            const float y = (iy + 0.5f) / PhotonDiv;
            float z = 3.0f;


            if (normalScale > 1)
            {

                int IX = x * 16;
                int IY = y * 16;

                utility::random::PCG_64_32 rng(IX + IY * 16);
                rng.next01();


                z += (rng.next01() * 2 - 1) * 3 * (normalScale / 5.0f);
            }


            const bool into = false; // レイがオブジェクトから出るのか、入るのか

            utility::Float3 org(x, y, z);
            utility::Float3 dir(0, 0, -1);

            auto normal = 1.0f * normalImage.loadBilinear(x, y);
            normal[2] = 1.0f;
            normal = normalize(normal);
            const auto orienting_normal = -normal;

            // Snellの法則
            const float nc = 1.0; // 真空の屈折率
            const float nt = kIor; // オブジェクトの屈折率
            const float nnt = into ? nc / nt : nt / nc;
            const float ddn = dot(dir, normal);
            const float cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

            /*
            if (cos2t < 0.0) { // 全反射
                incoming_radiance = radiance(reflection_ray, rnd, depth + 1);
                weight = now_object.color / russian_roulette_probability;
                break;
            }
            */
            // 屈折の方向
            const auto refraction_dir =
                normalize(dir * nnt - normal * (ddn * nnt + sqrt(cos2t)));

            float t = -org[2] / refraction_dir[2];

            const auto hitP = org + t * refraction_dir;

            const float u = hitP[0] / PhysicalSizePlane;
            const float v = hitP[1] / PhysicalSizePlane;

            const float C = scale * accumImageSize * accumImageSize / (PhotonDiv * PhotonDiv);

            accumImage.accumBilinear(u, v, { C, C, C });
        }
    }

    return accumImage;
}

#define BASE_PATH "C:/Code/VSProjects/akari8"

int goalBasedCaustics(int argc, char** argv)
{
#if 1
    constexpr float PhysicalSize = 1.0f;
    constexpr float eta = 1.5f;
    constexpr int hImageWidth = 128;

    const char* input{ BASE_PATH"/test05.hdr" };

    auto inputImage = utility::loadHDRImage(input);
    printf("%d, %d\n", inputImage.width, inputImage.height);
    const float averageValue = average(inputImage);
    printf("averageValue: %f\n", averageValue);

    utility::Image D(inputImage.width, inputImage.height);

    for (int iy = 0; iy < D.height; ++iy)
    {
        for (int ix = 0; ix < D.width; ++ix)
        {
            const float targetValue = inputImage.load(ix, iy)[0];
            const float diff = averageValue - targetValue;
            D.store(ix, iy, { diff, diff , diff });
        }
    }

    constexpr int Sub = 8;
    std::vector<Point> points;
    for (int iy = 0; iy < inputImage.height; ++iy)
    {
        for (int ix = 0; ix < inputImage.width; ++ix)
        {
            for (int sy = 0; sy < Sub; ++sy)
            {
                for (int sx = 0; sx < Sub; ++sx)
                {
                    Point pt;
                    pt.x = PhysicalSize * ((ix + (sx + 0.5f) / Sub) / inputImage.width);
                    pt.y = PhysicalSize * ((iy + (sy + 0.5f) / Sub) / inputImage.height);
                    pt.u = pt.x;
                    pt.v = pt.y;
                    points.push_back(pt);
                }
            }
        }
    }

    float DT = 1.0f;
    constexpr int kIteration = 1;
    for (int iteration = 0; iteration < kIteration; ++iteration)
    {
        char buf[256];
        printf("[%d]\n", iteration);

        printf("solve\n");
        auto phiImage = gauss_sidel(D, 1e-5f, 500000, PhysicalSize / D.width, 1.0f);
        sprintf(buf, BASE_PATH"/phi_%02d.hdr", iteration);
        utility::writeHDRImage(buf, luminanceToRedBlue(phiImage));

        sprintf(buf, BASE_PATH"/D_%02d.hdr", iteration);
        utility::writeHDRImage(buf, luminanceToRedBlue(D));

        printf("advect\n");
        advect(phiImage, PhysicalSize, PhysicalSize / phiImage.width, points, DT);
        DT *= 0.5f;

        sprintf(buf, BASE_PATH"/advectedPoints_%02d.hdr", iteration);
        savePoints(buf, points, 128, PhysicalSize);

        const auto mapping = createMapping(points, hImageWidth, PhysicalSize);
        sprintf(buf, BASE_PATH"/mapping_%02d.hdr", iteration);
        utility::writeHDRImage(buf, mapping);

        const auto mappingValidation = applyMapping(mapping, PhysicalSize);
        sprintf(buf, BASE_PATH"/mappingValidation_%02d.hdr", iteration);
        utility::writeHDRImage(buf, mappingValidation);

        // create normal map
        utility::Image hImage(hImageWidth, hImageWidth);

        utility::Image div_of_Nxy;
        utility::Image normalXY;

        for (int i = 0; i < 1; ++i)
        {
            normalXY = computeNormalXY(points, mapping, hImage, 3.0f, eta, PhysicalSize);

            sprintf(buf, BASE_PATH"/normalXY%02d.hdr", iteration);
            utility::writeHDRImage(buf, normalXY);

            // utility::writeHDRImage(BASE_PATH"/hImage2.hdr", luminanceToRedBlue(hImage));
            div_of_Nxy = diversity(normalXY, PhysicalSize / normalXY.width, -1.0f);
            hImage = gauss_sidel(div_of_Nxy, 1e-4f, 500000, PhysicalSize / div_of_Nxy.width, 1.0f);
        }
        // utility::writeHDRImage(BASE_PATH"/hImage2.hdr", luminanceToRedBlue(hImage));

#if 0
        // DEBUG
        for (int iy = 0; iy < hImage.height; ++iy)
        {
            for (int ix = 0; ix < hImage.width; ++ix)
            {
                hImage.store(ix, iy, {});
            }
        }
#endif

        utility::Image normalImage = gradient(hImage, PhysicalSize / hImage.width);

        /*
        for (int iy = 0; iy < normalImage.height; ++iy)
        {
            for (int ix = 0; ix < normalImage.width; ++ix)
            {
                normalImage.store(ix, iy, normalImage.load(ix, iy));
            }
        }

        utility::writeHDRImage(BASE_PATH"/normalImage.hdr", normalImage);
        */

        auto photonImage = generatePhotonImage(points, /*normalImage*/ normalXY, averageValue, 1.0f);
        sprintf(buf, BASE_PATH"/photonImage_%02d.hdr", iteration);
        utility::writeHDRImage(buf, photonImage);

        sprintf(buf, BASE_PATH"/normalXY%02d.dump", iteration);
        utility::dumpHDRImage(buf, normalXY);
#if 0
        if (iteration == 0)
        {
            for (int i = 0; i < 10; ++i)
            {

                auto photonImage2 = generatePhotonImage(points, /*normalImage*/ normalXY, averageValue, 1.0f + (9.0f - i));
                sprintf(buf, BASE_PATH"/photonImage2_%02d.hdr", i);
                utility::writeHDRImage(buf, photonImage2);
            }
        }
#endif

#endif

        // next D
        for (int iy = 0; iy < D.height; ++iy)
        {
            for (int ix = 0; ix < D.width; ++ix)
            {
                const float targetValue = inputImage.load(ix, iy)[0];
                const float diff = photonImage.load(ix, iy)[0] - targetValue;
                D.store(ix, iy, { diff, diff , diff });
            }
        }
    }

    return 0;
}

struct Camera
{
    utility::Float3 org;
    utility::Float3 dir;
    utility::Float3 up;

    float distanceToFilm;
    float filmWidth;
    float filmHeight;
};

struct Parameter
{
    int width{ 1024 };
    int height{ 1024 };
};


// このu, vは[-1, 1]
utility::Ray
generateCameraRay(const Camera& camera, float u, float v)
{
    const auto side = normalize(cross(camera.up, camera.dir));
    const auto up = normalize(cross(side, camera.dir));

    const auto p_on_film = camera.org + camera.distanceToFilm * camera.dir +
        side * u * camera.filmWidth / 2.0f +
        up * v * camera.filmHeight / 2.0f;

    const auto dir = normalize(p_on_film - camera.org);
    
    return { camera.org, dir };
}


struct Tree
{
    using Ite = std::vector<utility::Box*>::iterator;

    struct Node
    {
        utility::AABB aabb;
        std::unique_ptr<Node> left{nullptr};
        std::unique_ptr<Node> right{nullptr};
        std::array<utility::Box*, 4> box{};
    };

    std::unique_ptr<Node> create_(std::vector<utility::Box*>& boxs, Ite b, Ite e)
    {
        std::unique_ptr<Node> current = std::make_unique<Node>();

        utility::AABB aabb;
        for (Ite ite = b; ite != e; ++ite)
        {
            aabb.merge((*ite)->aabb);
        }
        current->aabb = aabb;

        if ((e - b) <= 4)
        {
            int i = 0;
            for (Ite ite = b; ite != e; ++ite)
            {
                current->box[i] = *ite;
                ++i;
            }
            return current;
        }

        utility::Float3 len = aabb.bounds[1] - aabb.bounds[0];

        int axis = 0;
        float l = -1;
        for (int a = 0; a < 3; ++a) {
            if (len[a] > l)
            {
                l = len[a];
                axis = a;
            }
        }

        auto med = (e - b) / 2 + b;
        std::nth_element(b, med, e, [axis](auto& a, auto& b)
            {
                return a->center[axis] < b->center[axis];
            });

        current->left = create_(boxs, b, med);
        current->right = create_(boxs, med, e);
        return current;
    }

    void clear() 
    {
        root = {};
    }


    std::unique_ptr<Node> root;
    void create(std::vector<utility::Box*>& boxs)
    {
        root = create_(boxs, boxs.begin(), boxs.end());
    }

    bool traverseLeaf_(const std::unique_ptr<Node>& node, const utility::Ray& ray, utility::Hitpoint& hitpoint) const
    {
        bool intersected = false;
        for (auto& b : node->box)
        {
            if (!b)
            {
                continue;
            }

            const auto tmphp = b->intersect(ray);
            if (tmphp && tmphp->distance < hitpoint.distance)
            {
                intersected = true;
                hitpoint = *tmphp;
            }
        }
        return intersected;
    }

    bool traverse_(const std::unique_ptr<Node>& node, const utility::Ray& ray, utility::Hitpoint& hitpoint) const
    {
        if (!node->left && !node->right)
        {
            // leaf
            return traverseLeaf_(node, ray, hitpoint);
        }


        std::optional<utility::Hitpoint> tmphp0;
        std::optional<utility::Hitpoint> tmphp1;
        if (node->left)
        {
            tmphp0 = node->left->aabb.intersect(ray);
        }

        if (node->right)
        {
            tmphp1 = node->right->aabb.intersect(ray);
        }

        std::unique_ptr<Node>* n0 = &node->left;
        std::unique_ptr<Node>* n1 = &node->right;

        if (tmphp0 && tmphp1 && tmphp0->distance >= tmphp1->distance)
        {
            std::swap(n0, n1);
            std::swap(tmphp0, tmphp1);
        }

        bool intersected = false;

        if (tmphp0 && hitpoint.distance > tmphp0->distance)
        {
            utility::Hitpoint tmphp;
            if (traverse_(*n0, ray, tmphp))
            {
                if (hitpoint.distance > tmphp.distance)
                {
                    intersected = true;
                    hitpoint = tmphp;
                }
            }
        }

        if (tmphp1 && hitpoint.distance > tmphp1->distance)
        {
            utility::Hitpoint tmphp;
            if (traverse_(*n1, ray, tmphp))
            {
                if (hitpoint.distance > tmphp.distance)
                {
                    intersected = true;
                    hitpoint = tmphp;
                }
            }
        }

        return intersected;
    }

    bool intersect(const utility:: Ray& ray, utility::Hitpoint& hitpoint) const {
        return traverse_(root, ray, hitpoint);
    }
};

std::vector<utility::Box> g_box;
Tree g_tree;

std::vector<utility::Material> g_materials;

utility::Image g_normalXY;


constexpr int tmpPhotonTextureCount = 32;
constexpr int photonTextureReso = 512;
std::vector<utility::Image> g_tmpPhotonTexture;
utility::Image g_photonTexture;

void initializeAtFirst()
{
//    g_normalXY = utility::readDumpedHDRImage(BASE_PATH"/out/build/x64-release/normalXY00.dump");
    g_normalXY = utility::readDumpedHDRImage("./normalXY00.dump");
    g_tmpPhotonTexture.resize(tmpPhotonTextureCount);
}

const float g_movieTime = 10.0f; // 10秒

void createScene(float currentTime)
{
    g_box.clear();
    g_tree.clear();
    g_materials.clear();
    g_photonTexture = utility::Image(photonTextureReso, photonTextureReso);
    for (auto& t : g_tmpPhotonTexture)
    {
        t = utility::Image(photonTextureReso, photonTextureReso);
    }

    utility::random::PCG_64_32 rng;

    utility::Material material;
    material.type = utility::Material::Glass;
    g_materials.push_back(material);

    material.type = utility::Material::Diffuse;
    g_materials.push_back(material);


    /*
    constexpr float S = 2.0f;
    for (int i = 0; i < 50; ++i)
    {
        const float x = rng.next(-S, S);
        const float y = rng.next(0, 2 * S);
        const float z = rng.next(-S, S);

        const utility::Matrix3x3 rotX = utility::createRotX(rng.next01() * PI);
        const utility::Matrix3x3 rotY = utility::createRotY(rng.next01() * PI);
        const utility::Matrix3x3 rotZ = utility::createRotZ(rng.next01() * PI);
        const utility::Matrix3x3 rot = mult(mult(rotX, rotY), rotZ);

        const float s = 0.2f;
        g_box.push_back(utility::Box(utility::Float3(x, y, z), utility::Float3(s, s, s), rot, &g_materials[0]));
    }
    */


    /*
    // わりといいかんじ
    
    const float U = 1.0f - currentTime / g_movieTime;
    for (int i = 0; i < 10; ++i)
    {
        const float y = i / 10.0f * 1.0f + 2.0f;

        const utility::Matrix3x3 rotX = utility::createRotX(rng.next01() * PI * U);
        const utility::Matrix3x3 rotY = utility::createRotY(rng.next01() * PI * U);
        const utility::Matrix3x3 rotZ = utility::createRotZ(rng.next01() * PI * U);
        const utility::Matrix3x3 rot = mult(mult(rotX, rotY), rotZ);

        const float s = 0.2f;
        g_box.push_back(utility::Box(utility::Float3(0, y, 0), utility::Float3(0.5f, 0.1f, 0.5f), rot, &g_materials[0]));
    }
    (g_box.end() - 1)->enableSpecialEffect = true;
    */

    /*
    const float U = 0;
    for (int i = 0; i < 10; ++i)
    {
        const float y = i / 10.0f * 1.0f + 3.0f;

        const utility::Matrix3x3 rotX = utility::createRotX(rng.next01() * PI * U);
        const utility::Matrix3x3 rotY = utility::createRotY(rng.next01() * PI * U);
        const utility::Matrix3x3 rotZ = utility::createRotZ(rng.next01() * PI * U);
        const utility::Matrix3x3 rot = mult(mult(rotX, rotY), rotZ);

        const float s = 0.2f;
        auto b = utility::Box(utility::Float3(0, y, 0), utility::Float3(0.5f, 0.05f, 0.5f), rot, &g_materials[0]);

        if (i == 0)
        {
            b.enableSpecialEffect = true;
        }

        g_box.push_back(b);
    }
    */

    float X = 1.0f - utility::clampValue(1.0f - currentTime / 8.0f, 0.0f, 1.0f);

    if (X > 0.8f)
    {
        X = utility::easeInExpo((X - 0.8f) / 0.2f) * 0.2f + 0.8f;
    }

    const float U = 1.0f - X;
    for (int i = 0; i < 10; ++i)
    {
        float YOffset = 0;

        const float phase = i / 10.0f;
        constexpr float TM = 1.5f;
        const float T = utility::clampValue(currentTime- phase, 0.0f, TM);
        if (T < TM)
        {
            YOffset = 5 - 5 * utility::easeOutBack(T / TM);
        }



        const float y = 2 * i / 10.0f * 1.05f + 3.0f + YOffset;

        const utility::Matrix3x3 rotX = utility::createRotX(rng.next01() * PI * U);
        const utility::Matrix3x3 rotY = utility::createRotY(rng.next01() * PI * U * (i % 3 == 0 ? 2 : 1));
        const utility::Matrix3x3 rotZ = utility::createRotZ(rng.next01() * PI * U * (i % 5 == 0 ? 2 : 1));

        const int K = rng.next() % 3;
        const utility::Matrix3x3 rot =

            K == 0 ?  mult(mult(rotX, rotY), rotZ) :
           (K == 1 ?  mult(mult(rotZ, rotX), rotY) : 
                      mult(mult(rotZ, rotY), rotX));

        const float s = 0.2f;
        auto b = utility::Box(utility::Float3(0, y, 0), utility::Float3(0.5f, 0.05f, 0.5f), rot, &g_materials[0]);

        if (i == 0)
        {
            b.enableSpecialEffect = true;
        }

        g_box.push_back(b);
    }

    /*
    const float U = 1.0f - currentTime / g_movieTime;
    const utility::Matrix3x3 rotX = utility::createRotX(rng.next01() * PI * U);
    const utility::Matrix3x3 rotY = utility::createRotY(rng.next01() * PI * U);
    const utility::Matrix3x3 rotZ = utility::createRotZ(rng.next01() * PI * U);
    const utility::Matrix3x3 rot = mult(mult(rotX, rotY), rotZ);
    auto b = utility::Box(utility::Float3(0.0f, 3.0f, 0), utility::Float3(0.5f, 0.1f, 0.5f), rot, &g_materials[0]);
    b.enableSpecialEffect = true;
    g_box.push_back(b);
    */
    

    /*
    constexpr float S = 2.0f;
    for (int i = 0; i < 1000; ++i)
    {
        const float x = rng.next(-S, S);
        const float y = rng.next(0, 2 * S);
        const float z = rng.next(-S, S);

        const utility::Matrix3x3 rotX = utility::createRotX(rng.next01() * PI);
        const utility::Matrix3x3 rotY = utility::createRotY(rng.next01() * PI);
        const utility::Matrix3x3 rotZ = utility::createRotZ(rng.next01() * PI);
        const utility::Matrix3x3 rot = mult(mult(rotX, rotY), rotZ);

        const float s = 0.1f;
        g_box.push_back(utility::Box(utility::Float3(x, y, z), utility::Float3(s, s, s), rot, &g_materials[0]));
    }
    */

    g_box.push_back(utility::Box(utility::Float3(0, -1, 0), utility::Float3(4, 1, 4), utility::createIdentity(), &g_materials[1]));

    std::vector<utility::Box*> boxptr;
    for (auto& b : g_box)
    {
        boxptr.push_back(&b);
    }

    g_tree.create(boxptr);
}



std::optional < utility::Hitpoint >
checkIntersect(const utility::Ray& ray)
{
    utility::Hitpoint hp;
    if (!g_tree.intersect(ray, hp))
    {
        return {};
    }
    return hp;
    

    /*
    utility::Hitpoint hp;


    for (auto& b : g_box)
    {
        auto tmp = b.intersect(ray);
        if (!tmp) continue;
        if (tmp->distance < hp.distance)
            hp = *tmp;
    }
    if (hp.distance == utility::kLarge)
        return { };

    return hp;
    */
}

void print(const utility::Matrix3x3& mat)
{
    for (int i = 0; i < 9; ++i)
    {
        printf("%f,", mat.m[i]);
        if ((i + 1) % 3 == 0)
            printf("\n");
    }
}

using Rng = utility::random::PCG_64_32;


utility::Float3
convertWolrdPositionToPhotonTextureUV(const utility::Float3& pos)
{
    if (abs(pos[1]) > 0.01f)
    {
        return {-100, -100, 0};
    }

    const auto tu = (pos[0] + 4.0f) / 8.0f;
    const auto tv = (pos[2] + 4.0f) / 8.0f;
    return { tu, tv, 0.0f };
}

template<bool GatherPhoton = false>
utility::Float3 radiance(const utility::Ray& orgRay, float kIor, const utility::Color& light , Rng& rng, int tmpIndex = 0)
{
    utility::Float3 L(1, 1, 1);
    utility::Float3 BG(0.1f, 0.1f, 0.1f);
    utility::Float3 contribution(0, 0, 0);

    utility::Ray currentRay = orgRay;

    int insideCounter = 0;

    int bounce;
    constexpr int kMaxBounce = 40;
    for (bounce = 0; bounce < kMaxBounce; ++bounce)
    {
        const auto hp = checkIntersect(currentRay);
        if (!hp)
        {
            break;
        }

        const auto* primitive = hp->primitive;
        const auto* material = primitive ? primitive->getMaterial() : nullptr;
        if (!material)
        {
            break;
        }

        if (material->type == utility::Material::Diffuse)
        {
            const auto tuv = convertWolrdPositionToPhotonTextureUV(hp->position);
            /*
            const auto tu = (hp->position[0] + 0.5f);
            const auto tv = (hp->position[2] + 0.5f);
            */

            if (!GatherPhoton)
            {
                const float u = rng.next01();
                const float v = rng.next01();
                const auto nextDir = utility::sample_cos_weighted(u, v, hp->normal);
                const auto nextOrg = currentRay.org + currentRay.dir * hp->distance;

                if (0 <= tuv[0] && tuv[0] <= 1.0f &&
                    0 <= tuv[1] && tuv[1] <= 1.0f)
                {
                    const auto emission = g_photonTexture.loadBilinear(tuv[0], tuv[1]);
                    if (length2(emission) > 0)
                    {
                        contribution += product(emission, L);
                    }
                }

                const float rho = 0.05f;
                L = rho * L;
                currentRay.dir = nextDir;
                currentRay.org = nextOrg + nextDir * 0.001f;
            }
            else
            {
                // photon 収拾
                g_tmpPhotonTexture[tmpIndex].accumBilinear(tuv[0], tuv[1], product(L, light));
                return {};
            }
        }
        else if (material->type == utility::Material::Glass)
        {
            auto normal = hp->normal;
            
            if (hp->enableTex)
            {
                auto t = -g_normalXY.loadBilinear(hp->uv[0], hp->uv[1]);
                utility::Float3 tmpnormal;
                normal[0] = t[0];
                normal[1] = -1.0f;
                normal[2] = t[1];
                normal = normalize(normal);
                const auto& mat = primitive->getMatrix();
                normal = utility::applyMatrix(mat, normal);
            }
            
            const utility::Float3 orienting_normal =
                dot(normal, currentRay.dir) < 0.0 ? normal : -normal;

            const utility::Ray reflectionRay =
                utility::Ray(hp->position, currentRay.dir - normal * 2.0f * dot(normal, currentRay.dir));


            const bool into = dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

            if (into)
            {
                insideCounter++;
            }
            else
            {
                insideCounter--;
            }

            bool inside = false;
            for (auto& b : g_box) {
                if (b.inside(currentRay.org)) {
                    inside = true;
                    break;
                }
            }
            if (inside)
            {
                if (into)
                {
                    insideCounter = 1;
                }
            }

            float nc = 1.0f;
            float nt = kIor;
            if (insideCounter == 1 && !into)
            {
                nc = kIor;
            }
            else if (insideCounter >= 2)
            {
                nc = kIor;
            }
            
#if 0
            // insideを呼ぶタイプの実装～
            const bool into = dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

            bool inside = false;

            for (auto& b : g_box)
            {
                if (b.inside(currentRay.org))
                {
                    inside = true;
                    break;
                }
            }

            bool inside2 = false;
            if (inside)
            {
                const auto o = hp->position + 0.001f * currentRay.dir;
                for (auto& b : g_box)
                {
                    if (b.inside(o))
                    {
                        inside2 = true;
                        break;
                    }
                }

            }


            float nc = 1.0f;
            float nt = kIor;

            if (inside2 && into)
            {
                nc = kIor;
            }
#endif



            /*
            const float nc = 1.0; // 真空の屈折率
            const float nt = kIor; // オブジェクトの屈折率
            */
            const float nnt = into ? nc / nt : nt / nc;
            const float ddn = dot(currentRay.dir, orienting_normal);
            const float cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

            if (cos2t < 0.0) { // 全反射
                currentRay.dir = reflectionRay.dir;
                currentRay.org = reflectionRay.org + reflectionRay.dir * 0.001f;
                continue;
            }

            // 屈折の方向
            const utility::Ray refractionRay = 
                utility::Ray(hp->position, normalize(currentRay.dir * nnt - normal * (into ? 1.0f : -1.0f) * (ddn * nnt + sqrt(cos2t))));

            const float a = nt - nc, b = nt + nc;
            const float R0 = (a * a) / (b * b);
            const float c = 1.0 - (into ? -ddn : dot(refractionRay.dir, -orienting_normal));
            const float Re = R0 + (1.0 - R0) * pow(c, 5.0);
            const float Tr = (1.0 - Re);

            //const float probability = GatherPhoton ? 0.0f : 0.25 + 0.5 * Re;
            const float probability = 0;

            if (rng.next01() < probability) { // 反射
                currentRay.dir = reflectionRay.dir;
                currentRay.org = reflectionRay.org + reflectionRay.dir * 0.001f;
                L = Re * L / probability;
            }
            else { // 屈折
                currentRay.dir = refractionRay.dir;
                currentRay.org = refractionRay.org + refractionRay.dir * 0.01f;
                L = Tr * L / (1 - probability);
            }
        }
    }

    if (bounce == kMaxBounce)
    {
        return {};
    }

    return product(L, BG) + contribution;
}


float luminance(const utility::Color& color)
{
    // assume sRGB color space
    return
        color[0] * 0.2126f +
        color[1] * 0.7152f +
        color[2] * 0.0722f;
}

void applyPostFX(utility::Image& image)
{
    printf("begin PostFx\n");

    utility::Image extract(image.width, image.height);

    const float kThreshold = 1.0f;
    const float kWeight = 0.1f;

#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.width; ++ix)
        {
            const auto col = image.load(ix, iy);
            if (luminance(col) > kThreshold)
            {
                extract.store(ix, iy, col);
            }
        }
    }

    // blur

    utility::Image tmp0 = extract;
    utility::Image tmp1(image.width, image.height);


    for (int i = 0; i < 1; ++i)
    {
#pragma omp parallel for schedule(dynamic, 1)
        for (int iy = 0; iy < image.height; ++iy)
        {
            for (int ix = 0; ix < image.width; ++ix)
            {
                const auto blured =
                    tmp0.load(ix - 3, iy) * 0.006f +
                    tmp0.load(ix - 2, iy) * 0.061f +
                    tmp0.load(ix - 1, iy) * 0.242f +
                    tmp0.load(ix, iy)     * 0.383f +
                    tmp0.load(ix + 1, iy) * 0.242f +
                    tmp0.load(ix + 2, iy) * 0.061f +
                    tmp0.load(ix + 3, iy) * 0.006f;
                tmp1.store(ix, iy, blured);
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int iy = 0; iy < image.height; ++iy)
        {
            for (int ix = 0; ix < image.width; ++ix)
            {
                const auto blured =
                    tmp1.load(ix, iy - 3) * 0.006f +
                    tmp1.load(ix, iy - 2) * 0.061f +
                    tmp1.load(ix, iy - 1) * 0.242f +
                    tmp1.load(ix, iy) * 0.383f +
                    tmp1.load(ix, iy + 1) * 0.242f +
                    tmp1.load(ix, iy + 2) * 0.061f +
                    tmp1.load(ix, iy + 3) * 0.006f;
                tmp0.store(ix, iy, blured);
            }
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.width; ++ix)
        {
            image.store(ix, iy, image.load(ix, iy) + kWeight * tmp0.load(ix, iy));
        }
    }

    printf("end PostFx\n");
}

bool kDebugOutput = false;

void renderSingleFrame(const Parameter& param, float currentTime, int frameNumber)
{
    const auto aspect = (float)param.width / param.height;

    Camera camera;

    camera.org = utility::Float3(10 - 2 * currentTime / 10.0f, 7 - currentTime/10.0f, 10 - 2 * currentTime / 10.0f);
    camera.dir = normalize(utility::Float3(0, 2 - currentTime / 10.0f, 0) - camera.org);
    camera.up = utility::Float3(0, 1, 0);

    camera.distanceToFilm = 1.0f;
    camera.filmHeight = 0.5f;
    camera.filmWidth = camera.filmHeight * aspect;

    utility::Image image(param.width, param.height);

    createScene(currentTime);

    float SCALE = 1;

    if (currentTime > 7)
    {
        SCALE = (1 - utility::clampValue((currentTime - 7) / 1.0f, 0.0f, 1.0f)) * 0.9f + 0.1f;
    }

    // photon
    constexpr int PhotonDiv = 512;
    {

        const float ior[8] = {
            1.485,
            1.490,
            1.495,
            1.5,
            1.505,
            1.510,
            1.515,
            1.520
        };

        const utility::Color lights[8] = {
            utility::Color(7.858960, 0.000000, 125.654218),
            utility::Color(0.000000, 21.924445, 101.874630),
            utility::Color(0.000000, 124.468216, 0.000000),
            utility::Color(139.134114, 86.750347, 0.000000),
            utility::Color(195.252931, 0.000000, 0.000000),
            utility::Color(25.281302, 0.000000, 0.000000),
            utility::Color(0.773990, 0.000000, 0.000000),
        };

        const auto Coeff = g_tmpPhotonTexture.size() / 32.0f * 0.25f;

        auto clampV = [](int v)
        {
            return utility::clampValue(v, 0, 7);
        };
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < g_tmpPhotonTexture.size(); ++i)
        {
            Rng rng;
            rng.set_seed(i);

            const int tmpIndex = i;
            for (int iy = 0; iy < PhotonDiv; ++iy)
            {
                for (int ix = 0; ix < PhotonDiv; ++ix)
                {
                    float fIndex = rng.next(0, 7);
                    int iIndex = (int)fIndex;
                    float fract = fIndex - iIndex;

                    const float IOR = (1 - fract) * ior[clampV(iIndex)] + fract*ior[clampV(iIndex + 1)];
                    const auto LIGHT = ((1 - fract) * lights[clampV(iIndex)] + fract * lights[clampV(iIndex + 1)]) / 100.0f * 0.01f * Coeff;


                    const float u = (ix + 0.5f) / PhotonDiv - 0.5f;
                    const float v = (iy + 0.5f) / PhotonDiv - 0.5f;

                    utility::Ray ray;
                    ray.org = utility::Float3(u, 50.0f, v);
                    ray.dir = utility::Float3(0, -1.0f, 0);

                    radiance<true>(ray, IOR, SCALE * LIGHT, rng, tmpIndex);
                }
            }
        }


        for (int iy = 0; iy < g_photonTexture.height; ++iy)
        {
            for (int ix = 0; ix < g_photonTexture.width; ++ix)
            {
                // reduce
                utility::Color c(0, 0, 0);
                for (int i = 0; i < g_tmpPhotonTexture.size(); ++i)
                {
                    c += g_tmpPhotonTexture[i].load(ix, iy);
                }

                c = c / (float)(PhotonDiv * PhotonDiv) * (float)(g_photonTexture.width * g_photonTexture.height);

                g_photonTexture.store(ix, iy, c);
            }
        }

        if (kDebugOutput && frameNumber == 1)
            utility::writeHDRImage(BASE_PATH"/render/g_photonTexture.hdr", g_photonTexture);
    }
    printf("generated\n");

    constexpr int S = kRough ? 1 : 2;

    const float kIor = 1.5f;

    const utility::Color kLight(1, 1, 1);

#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < image.height; ++iy)
    {
        Rng rng;
        rng.set_seed(iy);

        for (int ix = 0; ix < image.width; ++ix)
        {
            for (int sx = 0; sx < S; ++sx)
            {
                for (int sy = 0; sy < S; ++sy)
                {
                    const float u = -(ix + (sx + 0.5f) / S) / image.width * 2 + 1;
                    const float v = (iy + (sy + 0.5f) / S) / image.height * 2 - 1;
                    const auto ray = generateCameraRay(camera, u, v);
                    image.accum(ix, iy, radiance(ray, kIor, kLight, rng) * (1.0f / S));
                }
            }
        }
    }

    applyPostFX(image);

    if (kDebugOutput && frameNumber == 1)
        utility::writeHDRImage(BASE_PATH"/render/render.hdr", image);

    // hdr -> png
    constexpr int comp = 3;
    std::vector<uint8_t> ldrImage(image.width *  image.height * comp);

    float ScreenScale = 1;

    if (currentTime < 0.1f)
    {
        ScreenScale = 0;
    }

    if (currentTime < 0.6f)
    {
        ScreenScale = utility::easeOutQuad((currentTime - 0.1f) / 0.5f);
    }

    if (currentTime > 9.5f)
    {
        ScreenScale = 1.0f - utility::easeOutExpo((currentTime - 9.5f) / 0.5f);
    }

    for (int iy = 0; iy < image.height; ++iy)
    {
        for (int ix = 0; ix < image.width; ++ix)
        {
            const auto col = image.load(ix, iy);

            const uint8_t r{ (uint8_t)utility::clampValue(ScreenScale * pow(col[0], 1 / 1.8f) * 255, 0.0f, 255.0f) };
            const uint8_t g{ (uint8_t)utility::clampValue(ScreenScale * pow(col[1], 1 / 2.0f) * 255, 0.0f, 255.0f) };
            const uint8_t b{ (uint8_t)utility::clampValue(ScreenScale * pow(col[2], 1 / 2.2f) * 255, 0.0f, 255.0f) };

            const auto idx{ (ix + iy * image.width) * comp };

            ldrImage[idx + 0] = r;
            ldrImage[idx + 1] = g;
            ldrImage[idx + 2] = b;
        }
    }

    char buf[256];

    sprintf(buf, "./out/%03d.png", frameNumber);
    stbi_write_png(buf, image.width, image.height, comp, ldrImage.data(), image.width * sizeof(uint8_t)* comp);
}

int main(int argc, char** argv)
{
    Parameter param;

    if (kRough)
    {
        param.width /= 2;
        param.height /= 2;
    }

    initializeAtFirst();
    constexpr int maxFrameNumber = 240;

    //renderSingleFrame(param, 0, 1);
    
    for (int i = 0; i < maxFrameNumber; ++i)
    {
        printf("framenumber[%d]\n", i);
        renderSingleFrame(param, (i + 0.5f) / maxFrameNumber * g_movieTime, i + 1);
    }
}