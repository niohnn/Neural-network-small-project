#include "RefBridge.h"
#include "TestUtils.h"

#include <MathUtils.h>

using namespace dnntest;

TEST(MathUtils, DotProduct) {
    struct Case {
        std::string name;
        Vec a;
        Vec b;
    };
    std::vector<Case> cases = {
        {"empty", {}, {}},
        {"single element", {3.0}, {-2.0}},
        {"zero vector", {0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}},
        {"orthogonal", {1.0, 0.0}, {0.0, 1.0}},
        {"negative values", {-1.5, 2.5, -3.5}, {4.0, -5.0, 6.0}},
        {"mixed magnitudes", {1e10, 1e-10, 3.0}, {2e-5, 4e8, -1.0}},
        {"same vector", {0.25, -0.75, 1.5, 2.0}, {0.25, -0.75, 1.5, 2.0}},
    };
    std::mt19937 rng(1);
    for (const std::size_t size : {1u, 2u, 7u, 64u, 784u}) {
        cases.push_back({"random size " + std::to_string(size), randomVec(rng, size), randomVec(rng, size)});
    }

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(valueClose("dotProduct(a, b)", dnn::dotProduct(c.a, c.b), dnnref::dotProduct(c.a, c.b)));
    }
}

TEST(MathUtils, Sum) {
    struct Case {
        std::string name;
        Vec values;
    };
    std::vector<Case> cases = {
        {"empty", {}},
        {"single element", {42.5}},
        {"zeros", {0.0, 0.0, 0.0}},
        {"negative values", {-1.0, -2.5, -3.25}},
        {"cancelling values", {1.5, -1.5, 2.0, -2.0}},
        {"mixed magnitudes", {1e10, 3.0, 1e-10, -7.0}},
    };
    std::mt19937 rng(2);
    for (const std::size_t size : {1u, 3u, 10u, 128u, 784u}) {
        cases.push_back({"random size " + std::to_string(size), randomVec(rng, size, -10.0, 10.0)});
    }

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(valueClose("sum(values)", dnn::sum(c.values), dnnref::sum(c.values)));
    }
}
