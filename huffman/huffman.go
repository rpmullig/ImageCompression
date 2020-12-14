// utilities package for creating a huffman tree for use in huffman compression algorithm
package huffman

import (
	"image/color"
	"proj3/png"
	"proj3/minheap"
	"os"
	"encoding/binary"
	"fmt"
	"strings"
	"image"
	"math"
)

func GetFrequencyMap(img *png.Image) map[color.Color]int {
	
	freqmap := make(map[color.Color]int)
	
	var currentRGBAValue color.Color
	
	bounds := img.In.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
	
			currentRGBAValue = img.In.At(x, y)
	
			if _, exists := freqmap[currentRGBAValue]; exists {
				freqmap[currentRGBAValue]++
			} else {
				freqmap[currentRGBAValue] = 1
			}
		}
	}
	
	//fmt.Printf("Bounds: %d X %d", bounds.Max.Y, bounds.Max.X)
	
	return freqmap
}

func BuildHuffmanTree(frequencyMap map[color.Color]int) minheap.ColorNode {

	minHeap := minheap.NewMinHeap(len(frequencyMap))
	
	for k, v := range frequencyMap {
		minHeap.Insert(minheap.ColorNode{Frequency: v, ColorValue: k,
						Parent: nil, Right: nil, Left: nil})
	}
	minHeap.BuildMinHeap()
	
	for minHeap.Size > 1 {
		item_one := minHeap.Remove()
		item_two := minHeap.Remove()
		
		combinedFrequency := item_one.Frequency + item_two.Frequency
		
		newNode := minheap.ColorNode{Frequency: combinedFrequency, ColorValue: nil,
						Parent: nil, Right: &item_one, Left: &item_two}
		item_one.Parent = &newNode
		item_two.Parent = &newNode
		
		minHeap.Insert(newNode)
	}

	return minHeap.Remove()
}

func CompressFile(img *png.Image, huffmanCodeMap map[color.Color]string) int {

	var currentRGBAValue color.Color
	var code string
	
	bounds := img.In.Bounds()
	var bitStringBuilder strings.Builder

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
		
			currentRGBAValue = img.In.At(x, y)
			code = huffmanCodeMap[currentRGBAValue]
			bitStringBuilder.WriteString(code)	

		}
	}
	
	bitString :=  bitStringBuilder.String()

	lenB := len(bitString) / 8 + 1
    bs := make([]byte, lenB)

    count, i := 0,0
    var now byte

    for _, v := range bitString {
        if count == 8 {
            bs[i]=now
            i++
            now,count = 0,0
        }
        now = now << 1 + byte(v - '0')
        count++
    }
    if count!=0 {
        bs[i]= now <<  8-byte(count)
        i++
    }

    bs = bs[:i:i]

	f, _:= os.OpenFile("out.dat", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	binary.Write(f, binary.LittleEndian, bs) 
	f.Close()
	
	return lenB

}

func DecompressFile(huffmanColorMap map[string]color.Color, img *image.RGBA64, compressedLength int){

	f, err := os.Open("out.dat")

	if err != nil {
		panic(err)
	}


	output := make([]byte, compressedLength)

	binary.Read(f, binary.LittleEndian, &output)
	
	f.Close()

	var bitStringBuilder strings.Builder
	
	for _, bits := range output {
		s := fmt.Sprintf("%08b", bits) // careful to keep it of size 8 
		bitStringBuilder.WriteString(s)
	}	
	
	bitString :=  bitStringBuilder.String() 
	//fmt.Printf("\n\nDecompressed size %d bits", len(bitString))
	
	var tmpStr string
	var currentRGBAValue color.Color
	indx := 0 
	

	bounds := img.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {

			tmpStr = ""
			_, exists := huffmanColorMap[tmpStr]
			
			for !exists {
				tmpStr += string(bitString[indx])
				indx++
				_, exists = huffmanColorMap[tmpStr]
			}
			
			currentRGBAValue = huffmanColorMap[tmpStr]
			
			r, g, b, a := currentRGBAValue.RGBA()
			
			img.Set(x, y, color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)})
		}		
	}	
	
	png.Save(img, "out.png")
}

func convertBinaryToDecimal(number int) int {  
 decimal := 0  
 counter := 0.0  
 remainder := 0  
  
 for number != 0 {  
  remainder = number % 10  
  decimal += remainder * int(math.Pow(2.0, counter))  
  number = number / 10  
  counter++  
 }  
 return decimal  
}  
