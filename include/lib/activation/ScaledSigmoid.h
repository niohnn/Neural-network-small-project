#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {
    // Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
    class ScaledSigmoid : public ActivationFunction {
    public:
        ScaledSigmoid(double min, double max);
    
        double getMin() const;
        double getMax() const;
    
        virtual ~ScaledSigmoid()=default;
        
        virtual double activation(double x);
        virtual double activationBackwards(double x);
        virtual std::string getName() const;
    
    private:
        double min;
        double max;
    
    };

}
