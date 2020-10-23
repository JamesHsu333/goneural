package neural

import "math"

type ActivationType int

const (
	ActivationNone ActivationType = 0

	ActivationSigmoid ActivationType = 1
)

type Activation interface {
	F(float64) float64
	Df(float64) float64
}

func GetActivation(act ActivationType) Activation {
	switch act {
	case ActivationSigmoid:
		return Sigmoid{}
	}
}

func activation(act func(float64) float64) func(int, int, float64) float64 {
	return func(i, j int, v float64) float64 {
		return act(v)
	}
}

type Sigmoid struct{}

func (s Sigmoid) F(x float64) float64 {
	return 1 / (1 + math.Exp(-1*x))
}

func (s Sigmoid) Df(x float64) float64 {
	return 1 / (1 + math.Exp(-1*x)) * (1 - (1 / (1 + math.Exp(-1*x))))
}
