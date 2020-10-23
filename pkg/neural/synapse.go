package neural

import (
	"gonum.org/v1/gonum/mat"
)

type Synapse struct {
	Weight *mat.Dense
}

func NewSynapse(in, out int, weight WeightRandomType) *Synapse {
	return &Synapse{
		Weight: mat.NewDense(out, in, randomArray(out*in, weight)),
	}
}

func randomArray(size int, weight WeightRandomType) []float64 {
	array := make([]float64, size)
	for i, _ := range array {
		array[i] = weight()
	}
	return array
}
