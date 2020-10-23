package neural

import (
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	A       ActivationType
	Neurons *mat.Dense
}

func NewLayer(n int, activation ActivationType) *Layer {
	return &Layer{
		A:       activation,
		Neurons: mat.NewDense(n, 1, nil),
	}
}

type Bias struct {
	Neurons *mat.Dense
}

func NewBias(n int) *Bias {
	return &Bias{
		Neurons: mat.NewDense(n, 1, nil),
	}
}
