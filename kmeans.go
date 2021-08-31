package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

type ShapeError struct {
	info string
}

func (e *ShapeError) Error() string {
	return e.info
}

func NewShapeError(info string) *ShapeError {
	return &ShapeError{
		info: info,
	}
}

//read data from file,return the data 1-D array,and rows info,cols info
func ReadSamples(filePath string) ([]float64, int, int) {
	f, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	//read the rows and cols info
	line, err := reader.ReadString('\n')
	if err != nil {
		panic(err)
	}
	line = line[:len(line)-1]
	//split our line use sep
	lineSplit := strings.Split(line, " ")
	rows, err := strconv.Atoi(lineSplit[0])
	if err != nil {
		panic(err)
	}
	cols, err := strconv.Atoi(lineSplit[1])
	if err != nil {
		panic(err)
	}
	log.Printf("reading matrix with shape %d x %d\n", rows, cols)
	//initialize our data with 1-D array
	data := make([]float64, rows*cols)
	var count int = 0
	var offset int = count * cols
	//read data by line
	for {
		line, err = reader.ReadString('\n')
		//when the line is empty
		if err == io.EOF {
			break
		}
		//traceback the read data error
		if err != nil {
			panic(err)
		}
		line = line[:len(line)-1]
		lineSplit = strings.Split(line, " ")

		for i := 0; i < cols; i++ {
			v, err := strconv.ParseFloat(lineSplit[i], 64)
			if err != nil {
				panic(err)
			}
			idx := offset + i
			data[idx] = v
		}
		count++
		offset = count * cols
	}
	return data, rows, cols
}

//compute the norm value,you can specify a concurrent number
func ComputeRowNorm(data []float64, rows int, cols int, concurrentNum int, norm float64) []float64 {
	size := len(data)
	if size != (rows * cols) {
		info := fmt.Sprintf("shape not match,you have %d data,but with shape %d x %d\n", size, rows, cols)
		panic(NewShapeError(info))
	}

	//to avoid you specify too many routine,will cost the extra time,each routine loop 100 at least
	if concurrentNum*100 > rows {
		concurrentNum = int(math.Ceil(float64(rows) / 100))
		// log.Printf("we adjust your routine number -> %d\n", concurrentNum)
	}

	rowNorm := make([]float64, rows)
	//use this flag to judge the right batch_size
	remainFlag := rows % concurrentNum
	var batchSize int
	if remainFlag > 0 {
		batchSize = rows / (concurrentNum + 1)
	} else {
		batchSize = rows / concurrentNum
	}
	// log.Printf("use %d concurrent to compute norm,each batch_size is %d\n", concurrentNum, batchSize)

	taskFunc := func(left int, right int, ch chan int) {
		for i := left; i < right; i++ {
			offset := i * cols
			var value float64 = 0.0
			for j := 0; j < cols; j++ {
				idx := offset + j
				value = value + math.Pow(data[idx], norm)
			}
			rowNorm[i] = value
		}
		// a simple way to join our routine
		ch <- 0
	}

	var left int = 0
	var right int = batchSize
	//make a buffer channel
	ch := make(chan int, concurrentNum)
	for i := 0; i < concurrentNum; i++ {
		//check the index
		if right > rows {
			right = rows
		}
		go taskFunc(left, right, ch)
		//plus the offset /batch_size
		left = left + batchSize
		right = right + batchSize
	}
	//wait the routine
	for i := 0; i < concurrentNum; i++ {
		<-ch
	}
	return rowNorm
}

//choose our center by step
func ChooseCenterStep(data []float64, rows int, cols int, k int) []float64 {
	//we must be sure the last sample index <rows
	centers := make([]float64, k*cols)
	stepSize := rows / k
	centerLeft := 0
	centerRight := cols

	var chooseLeft int
	var chooseRight int

	for i := 0; i < k; i++ {
		chooseLeft = i * stepSize * cols
		chooseRight = chooseLeft + cols
		copy(centers[centerLeft:centerRight], data[chooseLeft:chooseRight])
		//update the index
		centerLeft += cols
		centerRight += cols
	}
	return centers

}

func ChooseCenterRand(data []float64, rows int, cols int, k int) []float64 {
	randIndex := MakeRandSlice(0, rows, k)
	fmt.Println("rand index slice:", randIndex)
	centers := make([]float64, k*cols)
	centerLeft := 0
	centerRight := cols
	for _, v := range randIndex {
		copy(centers[centerLeft:centerRight], data[v*cols:(v+1)*cols])
		centerLeft += cols
		centerRight += cols
	}
	return centers
}

//generate k rand number between start -> end
func MakeRandSlice(start int, end int, k int) []int {
	if (start > end) || (end-start) < k {
		panic(errors.New("illegal range error"))
	}
	result := make([]int, k)
	//use the timestamp as a rand seed
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	// set a flag to respect the number whether exist
	var flag bool = true
	//produce k random center index
	for i := 0; i < k; {
		//will contain the left,never the right
		num := r.Intn(end-start) + start

		//the privious size is i,so we should compare i times
		for j := 0; j < i; j++ {
			if result[j] == num {
				flag = false
			}
		}
		if flag {
			result[i] = num
			i++
		}
	}
	return result
}

//compare two center result
func CompareCentersDiff(oldCenters []float64, newCenters []float64, cols int, k int) float64 {
	n1 := len(oldCenters)
	n2 := len(newCenters)
	if n1 != cols*k || n2 != cols*k || n1 != n2 {
		info := fmt.Sprintf("the size of two centers not match...n1->%d n2->%d cols->%d k->%d", n1, n2, cols, k)
		panic(NewShapeError(info))
	}
	//compute the sub square of two centers
	var result float64 = 0.0
	for i := 0; i < k*cols; i++ {
		v1 := oldCenters[i]
		v2 := newCenters[i]
		result += math.Pow(v1-v2, 2)
	}
	result = math.Sqrt(result)
	result = result / float64(k)
	return result
}

func MatMul(m1 []float64, r1 int, c1 int, m2 []float64, r2 int, c2 int) []float64 {
	if c1 != c2 {
		panic(errors.New("the column of two matrix not equal"))
	}

	if len(m1) != (r1*c1) || len(m2) != (r2*c2) {
		panic(NewShapeError("size error..."))
	}

	indexCache := make([]int, r2)
	for j := 0; j < r2; j++ {
		indexCache[j] = j * c2
	}
	mulResult := make([]float64, r1*r2)
	//memory continuous
	for i := 0; i < r1; i++ {
		offset1 := i * c1
		mulOffset := i * r2
		for j := 0; j < r2; j++ {
			offset2 := indexCache[j]
			v := 0.0
			for k := 0; k < c2; k++ {
				idx1 := offset1 + k
				idx2 := offset2 + k
				v = v + m1[idx1]*m2[idx2]
			}
			// fmt.Println(v)
			mulResult[mulOffset+j] = v
		}
	}
	return mulResult
}

//fetch the transpose of the vecotr
func VectorTranspose(v1 []float64, r1 int, c1 int) []float64 {
	result := make([]float64, c1*r1)
	indexCache := make([]int, c1)
	for j := 0; j < c1; j++ {
		indexCache[j] = j * r1
	}
	for i := 0; i < r1; i++ {
		offset := i * c1
		for j := 0; j < c1; j++ {
			idx1 := offset + j
			idx2 := indexCache[j] + i
			result[idx2] = v1[idx1]
		}
	}
	return result
}

func SetZero(v []int) {
	n := len(v)
	for i := 0; i < n; i++ {
		v[i] = 0
	}
}

func PrintCenters(centers []float64, k int, cols int) {
	if k*cols != len(centers) {
		panic(errors.New("data size not match"))
	}
	for i := 0; i < k; i++ {
		offset := i * cols
		for j := 0; j < cols; j++ {
			fmt.Printf("%.5f ", centers[offset+j])
		}
		fmt.Println()
	}
}
func Kmeans(samples []float64, k int, rows int, cols int, concurrentNum int, normConcurrent int, init string, iters int, thresh float64) ([]int, []float64) {
	//compute the row 2 norm
	rowNorm := ComputeRowNorm(samples, rows, cols, normConcurrent, 2)
	//initialize labels vector
	labels := make([]int, rows)

	//initialize our centers
	var (
		centers    []float64
		centerDiff float64
	)
	oldCenters := make([]float64, k*cols)
	if init == "rand" {
		centers = ChooseCenterRand(samples, rows, cols, k)
	} else if init == "step" {
		centers = ChooseCenterStep(samples, rows, cols, k)
	}

	centerRowNorm := ComputeRowNorm(centers, k, cols, 1, 2)

	//define our train func
	trainFunc := func(left int, right int, sampleSumCh chan []float64, sampleNumCh chan []int) {
		subLeft := left * cols
		sampleSum := make([]float64, k*cols)
		sampleNum := make([]int, k)
		subRight := right * cols
		subRows := right - left
		subVector := samples[subLeft:subRight]
		// centersTran := VectorTranspose(centers, k, cols)
		mulResult := MatMul(subVector, subRows, cols, centers, k, cols)
		for i := 0; i < subRows; i++ {
			minIndex := 0
			offset := i * k
			previousDis := rowNorm[left+i] + centerRowNorm[0] - 2*mulResult[offset]
			for j := 1; j < k; j++ {
				distance := rowNorm[left+i] + centerRowNorm[j] - 2*mulResult[offset+j]
				if distance < previousDis {
					minIndex = j
					previousDis = distance
				}
			}
			labels[left+i] = minIndex
			sampleNum[minIndex]++
			for c := 0; c < cols; c++ {
				sampleSum[minIndex*cols+c] += subVector[minIndex*cols+c]
			}
		}
		sampleSumCh <- sampleSum
		sampleNumCh <- sampleNum
	}

	var (
		remainFlag int
		batchSize  int
		start      time.Time
		elapsed    time.Duration
	)
	remainFlag = rows % concurrentNum
	if remainFlag == 0 {
		batchSize = rows / concurrentNum
	} else {
		batchSize = rows / (concurrentNum - 1)
	}
	sampleSumCh := make(chan []float64, concurrentNum)
	sampleNumCh := make(chan []int, concurrentNum)
	kindSampleNum := make([]int, k)
	for iter := 0; iter < iters; iter++ {
		SetZero(kindSampleNum)
		start = time.Now()
		// fmt.Println(iter, centers)
		left := 0
		right := batchSize
		//join the routine and compute the new center
		copy(oldCenters, centers)

		for i := 0; i < concurrentNum; i++ {
			//check the index
			if right > rows {
				right = rows
			}
			go trainFunc(left, right, sampleSumCh, sampleNumCh)
			left += batchSize
			right += batchSize
		}

		for i := 0; i < concurrentNum; i++ {
			sampleSum := <-sampleSumCh
			for j := 0; j < k*cols; j++ {
				centers[j] += sampleSum[j]
			}
			sampleNum := <-sampleNumCh
			for j := 0; j < k; j++ {
				kindSampleNum[j] += sampleNum[j]
			}
		}

		for i := 0; i < len(centers); i++ {
			centers[i] = centers[i] - oldCenters[i]
		}
		//compute new center
		for i := 0; i < k; i++ {
			offset := i * cols
			for j := 0; j < cols; j++ {
				idx := offset + j
				centers[idx] = centers[idx] / (float64(kindSampleNum[i]) + 1e-5)
				centers[idx] = 0.2*oldCenters[idx] + 0.8*centers[idx]
			}
		}

		centerDiff = CompareCentersDiff(oldCenters, centers, cols, k)
		//if the center not change,break
		if centerDiff < thresh {
			log.Println("iteration at", iter, ",center changed less than", thresh)
			break
		}
		//update the center row 2 norm
		centerRowNorm = ComputeRowNorm(centers, k, cols, 1, 2)
		elapsed = time.Since(start)
		log.Println("iteration at ", iter+1, "cost time ", elapsed)
	}
	fmt.Println(kindSampleNum)
	return labels, centers
}

func main() {
	filePath := "test_data/test.txt"
	samples, rows, cols := ReadSamples(filePath)
	thresh := 0.01
	k := 20
	iters := 100
	concurrentNum := 20
	Kmeans(samples, k, rows, cols, concurrentNum, 1, "step", iters, thresh)
}
