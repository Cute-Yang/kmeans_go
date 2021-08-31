package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func a() {
	s1 := []float64{3, 1, 1.2, 4.1, 1.2, 0.8}
	m1 := mat.NewDense(3, 2, s1)
	fmt.Println(mat.Formatted(m1))
	v1 := make([]float64, 3*2)
	fmt.Println(v1)
}
