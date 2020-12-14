package main

import (
	"proj3/png"
	"os"
	"encoding/json"
	"proj3/huffman"
	"proj3/minheap"
	"fmt"
	//"encoding/binary"
	"image/color"
	"image"
	"strings"
	"flag"
	"os/exec"
	"strconv"
)

type Request struct {
	InPath string
}

type mapItem struct {
	code string
	value interface{}
}

func main(){

	decoder := json.NewDecoder(os.Stdin)
	var request Request
	decoder.Decode(&request) 
	img, err := png.Load(request.InPath)
	
	if err != nil {
		fmt.Errorf("Error in reading file %g", err)
		return 
	}
	
	frequencyMap := huffman.GetFrequencyMap(img)

	huffmanTreeRoot := huffman.BuildHuffmanTree(frequencyMap) 
	
	huffmanCodeMap := make(map[color.Color]string)
	huffmanColorMap := make(map[string]color.Color)
	
	InorderTraversal(huffmanTreeRoot, "", huffmanCodeMap, huffmanColorMap) 

	inBounds := img.In.Bounds()
	outImg := image.NewRGBA64(inBounds)

	inputThreadCount := flag.Int("p", 1, "Number of GPU threads")
	flag.Parse()
	outputFilename := strings.Split(request.InPath, ".")[0] + "_out.png"
	

	// Program in parallel will have to have standard lengths so 
	// there can be appropriate division of the data
	if *inputThreadCount > 1 {
		huffmanColorMap = standardizeMapSize(huffmanCodeMap, huffmanColorMap)
	}
	
	compressedLength := huffman.CompressFile(img, huffmanCodeMap) // in bytes 

	maxCodeLength := 0
	for _, v := range huffmanCodeMap{
		if maxCodeLength < len(v) {
			maxCodeLength = len(v)
		}
	}
	
	
	if *inputThreadCount == 1 {
		// sequential version 
		// does not need to have standard length
		huffman.DecompressFile(huffmanColorMap, outImg, compressedLength)
	} else {
			
		
		// storing the the map is needed in another coding language 
		// difference in speedup due to storing the map
		f, _ := os.Create("map.txt")
		for k, v := range huffmanCodeMap {
			r, g, b, a := k.RGBA()
			s := fmt.Sprintf("%s\n%d\n%d\n%d\n%d\n", v, r, g, b, a)
			f.Write([]byte(s))
		}
		f.Close()		
		
		cmd := exec.Command("gpu_program.exe", "map.txt", "out.dat", strconv.Itoa(inBounds.Max.Y), strconv.Itoa(inBounds.Max.X), strconv.Itoa(maxCodeLength), strconv.Itoa(*inputThreadCount), outputFilename)
		out, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Printf("\n%v", err)
		}
		fmt.Printf("%s\n", out)
		//stderr, _ := cmd.StderrPipe()
		//fmt.Printf("%s\n", stderr)
	}
}




func InorderTraversal(currentNode minheap.ColorNode, code string, huffmanCodeMap map[color.Color]string, huffmanColorMap map[string]color.Color) {

	if currentNode.Left != nil {
			InorderTraversal(*currentNode.Left, code+"1", huffmanCodeMap, huffmanColorMap)
	}
	
	if currentNode.Left == nil && currentNode.Right == nil {
		huffmanCodeMap[currentNode.ColorValue] = code 
		huffmanColorMap[code] = currentNode.ColorValue
		//fmt.Printf("Node %s with code of %s\n", currentNode.ColorValue, code)
	}


	if currentNode.Right != nil {
			InorderTraversal(*currentNode.Right, code+"0", huffmanCodeMap, huffmanColorMap)
	}
	
	return
}



func standardizeMapSize(huffmanCodeMap map[color.Color]string, huffmanColorMap map[string]color.Color) map[string]color.Color {

	maxCodeLength := 0
	for _, v := range huffmanCodeMap{
		if maxCodeLength < len(v) {
			maxCodeLength = len(v)
		}
	}
	//fmt.Printf("Longest code: %d" , maxCodeLength) 
	
	for maxCodeLength % 8 != 0 {
		fmt.Printf("\nCode length increased for byte alignment. Now it's %d bits\n", maxCodeLength) 		
		maxCodeLength++
	}


	prefix := strings.Repeat("0", maxCodeLength)

	newColorMap := make(map[string]color.Color)

	for k, v := range huffmanCodeMap{
		standardCodeStr := prefix[0: maxCodeLength - len(v)] + v
		huffmanCodeMap[k] = standardCodeStr
		newColorMap[standardCodeStr] = k 
	}
	
	return newColorMap

}
