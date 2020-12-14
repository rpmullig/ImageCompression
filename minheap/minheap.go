// modified code from https://golangbyexample.com/minheap-in-golang/
//
package minheap

import (
	"image/color"
	"fmt"
)

type ColorNode struct {
	Frequency int
	ColorValue color.Color
	Parent *ColorNode
	Right *ColorNode
	Left *ColorNode
}

type minheap struct {
    heapArray []ColorNode
    Size      int
    maxsize   int
}

func NewMinHeap(maxsize int) *minheap {
    minheap := &minheap{
        heapArray: []ColorNode{},
        Size:      0,
        maxsize:   maxsize,
    }
    return minheap
}

func (m *minheap) leaf(index int) bool {
    if index >= (m.Size/2) && index <= m.Size {
        return true
    }
    return false
}

func (m *minheap) parent(index int) int {
    return (index - 1) / 2
}

func (m *minheap) leftchild(index int) int {
    return 2*index + 1
}

func (m *minheap) rightchild(index int) int {
    return 2*index + 2
}

func (m *minheap) Insert(item ColorNode) error {
    if m.Size >= m.maxsize {
        return fmt.Errorf("Heal is ful")
    }
    m.heapArray = append(m.heapArray, item)
    m.Size++
    m.upHeapify(m.Size - 1)
    return nil
}

func (m *minheap) swap(first, second int) {
    temp := m.heapArray[first]
    m.heapArray[first] = m.heapArray[second]
    m.heapArray[second] = temp
}

func (m *minheap) upHeapify(index int) {
    for m.heapArray[index].Frequency < m.heapArray[m.parent(index)].Frequency {
        m.swap(index, m.parent(index))
        index = m.parent(index)
    }
}

func (m *minheap) downHeapify(current int) {
    if m.leaf(current) {
        return
    }
    smallest := current
    leftChildIndex := m.leftchild(current)
    rightRightIndex := m.rightchild(current)
    //If current is smallest then return
    if leftChildIndex < m.Size && m.heapArray[leftChildIndex].Frequency < m.heapArray[smallest].Frequency {
        smallest = leftChildIndex
    }
    if rightRightIndex < m.Size && m.heapArray[rightRightIndex].Frequency < m.heapArray[smallest].Frequency {
        smallest = rightRightIndex
    }
    if smallest != current {
        m.swap(current, smallest)
        m.downHeapify(smallest)
    }
    return
}
func (m *minheap) BuildMinHeap() {
    for index := ((m.Size / 2) - 1); index >= 0; index-- {
        m.downHeapify(index)
    }
}

func (m *minheap) Remove() ColorNode {
    top := m.heapArray[0]
    m.heapArray[0] = m.heapArray[m.Size-1]
    m.heapArray = m.heapArray[:(m.Size)-1]
    m.Size--
    m.downHeapify(0)
    return top
}