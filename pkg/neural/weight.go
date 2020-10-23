package neural

import "math/rand"

type WeightRandomType func() float64

func Uniform(stdDev, mean float64) WeightRandomType {
	return func() float64 {
		return (rand.Float64()-0.5)*stdDev + mean
	}
}

func Normal(stdDev, mean float64) WeightRandomType {
	return func() float64 {
		return rand.NormFloat64()*stdDev + mean
	}
}
