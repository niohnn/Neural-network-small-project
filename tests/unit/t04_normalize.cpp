#include "RefBridge.h"
#include "TestUtils.h"

#include <MathUtils.h>

using namespace dnntest;

TEST(Normalize, MatchesReference) {
    std::mt19937 rng(3);

    {
        SCOPED_TRACE("normalize(values) with int values");
        std::vector<std::vector<int>> cases = {
            {1, 2, 3}, {255, 0, 128}, {5}, {7, 7, 7}, {-3, 6, 0}, {-5, -2, -9}, {0, 1},
        };
        cases.push_back(randomInts(rng, 784, 0, 255));
        for (std::size_t i = 0; i < cases.size(); ++i) {
            SCOPED_TRACE("case " + std::to_string(i));
            EXPECT_TRUE(vecClose("normalize(values)", dnn::normalize(cases[i]), dnnref::normalize(cases[i])));
        }
    }
    {
        SCOPED_TRACE("normalize(values) with double values");
        // The maximum is used as an int, so non-integral maxima are truncated.
        std::vector<Vec> cases = {
            {1.0, 2.0, 4.0}, {0.5, 2.7, 1.0}, {3.0, 1.5}, {-2.0, 4.9}, {255.0, 17.5, 0.0}, {1.0},
        };
        cases.push_back(randomVec(rng, 50, 1.0, 100.0));
        for (std::size_t i = 0; i < cases.size(); ++i) {
            SCOPED_TRACE("case " + std::to_string(i));
            EXPECT_TRUE(vecClose("normalize(values)", dnn::normalize(cases[i]), dnnref::normalize(cases[i])));
        }
    }
    {
        SCOPED_TRACE("normalize(values, maxValue) with int values");
        const std::vector<int> values = randomInts(rng, 784, 0, 255);
        for (const int maxValue : {255, 1, 10, 1000, -4}) {
            SCOPED_TRACE("maxValue = " + std::to_string(maxValue));
            EXPECT_TRUE(vecClose("normalize(values, maxValue)", dnn::normalize(values, maxValue), dnnref::normalize(values, maxValue)));
            const std::vector<int> mixed = {-3, 0, 7, 12};
            EXPECT_TRUE(vecClose("normalize(mixed, maxValue)", dnn::normalize(mixed, maxValue), dnnref::normalize(mixed, maxValue)));
        }
    }
    {
        SCOPED_TRACE("normalize(values, maxValue) with double values");
        const Vec values = randomVec(rng, 784, 0.0, 255.0);
        for (const int maxValue : {255, 1, 7, -2}) {
            SCOPED_TRACE("maxValue = " + std::to_string(maxValue));
            EXPECT_TRUE(vecClose("normalize(values, maxValue)", dnn::normalize(values, maxValue), dnnref::normalize(values, maxValue)));
        }
        EXPECT_TRUE(vecClose("normalize({}, 255)", dnn::normalize(Vec{}, 255), dnnref::normalize(Vec{}, 255)));
    }
}
