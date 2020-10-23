package neural

import (
	"gonum.org/v1/gonum/mat"
)

type Neural struct {
	Layers   []*Layer
	Synapses []*Synapse
	Biases   []*Bias
	Config   *Config
}

type Config struct {
	Inputs       int
	Layers       []int
	Activation   ActivationType
	Weight       WeightRandomType
	LearningRate float64
}

func NewNeural(c *Config) *Neural {
	if c.Activation == ActivationNone {
		c.Activation = ActivationSigmoid
	}

	if c.Weight == nil {
		c.Weight = Normal(0.5, 0)
	}

	layers := initLayers(c)

	synapses := initSynapses(c)

	biaseses := initBiases(c)

	return &Neural{
		Layers:   layers,
		Synapses: synapses,
		Biases:   biaseses,
		Config:   c,
	}

}

func initLayers(c *Config) []*Layer {
	layers := make([]*Layer, len(c.Layers))
	for i, _ := range layers {
		layers[i] = NewLayer(c.Layers[i], c.Activation)
	}
	return layers
}

func initSynapses(c *Config) []*Synapse {
	synapses := make([]*Synapse, len(c.Layers))
	for i, _ := range synapses {
		if i == 0 {
			synapses[i] = NewSynapse(c.Inputs, c.Layers[i], c.Weight)
		} else {
			synapses[i] = NewSynapse(c.Layers[i-1], c.Layers[i], c.Weight)
		}
	}
	return synapses
}

func initBiases(c *Config) []*Bias {
	biases := make([]*Bias, len(c.Layers))
	for i, _ := range biases {
		biases[i] = NewBias(c.Layers[i])
	}
	return biases
}

func (n *Neural) Train(inputData [][]float64, epoch int) {

}

func (n *Neural) Feedforward(inputData []float64) {
	act := GetActivation(n.Config.Activation)
	input := mat.NewDense(len(inputData), 1, inputData)
	n.Layers[0].Neurons.Product(n.Synapses[0].Weight, input)
	for i := 1; i < len(n.Layers); i++ {
		n.Layers[i].Neurons.Product(n.Synapses[i].Weight, n.Layers[i-1].Neurons)
		n.Layers[i].Neurons.Apply(activation(act.F), n.Layers[i].Neurons)
	}
}

func (n *Neural) FindBias(targetsData []float64) {
	targets := mat.NewDense(len(targetsData), 1, targetsData)
	n.Biases[len(n.Biases)-1].Neurons.Sub(targets, n.Layers[len(n.Biases)-1].Neurons)
	for i := len(n.Biases) - 1; i > 0; i-- {
		n.Biases[i-1].Neurons.Product(n.Synapses[i].Weight, n.Biases[i].Neurons)
	}
}

func (n *Neural) BackPropagate() {
	for i := len(n.Synapses) - 1; i > 0; i-- {
	}
}
