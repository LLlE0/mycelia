package coordinator

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// ComputeCredit calculates the compute credit for a participant
// Formula: CC = 0.6*S(gpu) + 0.3*S(cpu) + 0.1*S(vram) + 0.2*CUDA_ENABLED
//
// Where:
// S(gpu) = FP(x)TFLOPS/100 + VRAM/12
// S(cpu) = Cores_number * CR * IPCrel/96
// S(vram) = min(1, VRAM/12)

//УПД
//Решено было отказаться от него: узел с кудой как-то по умолчанию обучающий
//Так что используется эта метрика сейчас только для мониторинга силы узлов, будет дорабатываться
type ComputeCredit struct {
	GPUInfo     GPUInfo `json:"gpu_info"`
	CPUInfo     CPUInfo `json:"cpu_info"`
	VRAMGB      float64 `json:"vram_gb"`
	CUDAEnabled bool    `json:"cuda_enabled"`

	S_GPU  float64 `json:"s_gpu"`
	S_CPU  float64 `json:"s_cpu"`
	S_VRAM float64 `json:"s_vram"`
	CC     float64 `json:"compute_credit"`
}

type GPUInfo struct {
	Name        string  `json:"name"`
	TFLOPS      float64 `json:"tflops"`  
	VRAMGB      float64 `json:"vram_gb"` 
	CUDACapable bool    `json:"cuda_capable"`
}

type CPUInfo struct {
	Model    string  `json:"model"`
	Cores    int     `json:"cores"`
	ClockMHz float64 `json:"clock_mhz"` 
	IPC      float64 `json:"ipc"`       
}

func (cc *ComputeCredit) Calculate(systemInfo map[string]interface{}) {
	if vram, ok := systemInfo["vram_gb"].(float64); ok && vram > 0 {
		cc.GPUInfo.VRAMGB = vram
	}
	if cuda, ok := systemInfo["cuda_available"].(bool); ok {
		cc.CUDAEnabled = cuda
		cc.GPUInfo.CUDACapable = cuda
	}

	if cc.GPUInfo.Name == "" && cc.GPUInfo.TFLOPS == 0 {
		if gpuObj, ok := systemInfo["gpu"].(map[string]interface{}); ok {
			if name, ok := gpuObj["name"].(string); ok {
				cc.GPUInfo = cc.analyzeGPU(name)
			}
			if vram, ok := gpuObj["vram_gb"].(float64); ok {
				cc.GPUInfo.VRAMGB = vram
			}
			if cuda, ok := gpuObj["cuda_available"].(bool); ok {
				cc.GPUInfo.CUDACapable = cuda
				cc.CUDAEnabled = cuda
			}
		} else if gpuStr, ok := systemInfo["gpu"].(string); ok {
			var gpuMap map[string]interface{}
			if err := json.Unmarshal([]byte(gpuStr), &gpuMap); err == nil {
				if name, ok := gpuMap["name"].(string); ok {
					cc.GPUInfo = cc.analyzeGPU(name)
				}
				if vram, ok := gpuMap["vram_gb"].(float64); ok {
					cc.GPUInfo.VRAMGB = vram
				}
				if cuda, ok := gpuMap["cuda_available"].(bool); ok {
					cc.GPUInfo.CUDACapable = cuda
					cc.CUDAEnabled = cuda
				}
			} else {
				cc.GPUInfo = cc.analyzeGPU(gpuStr)
			}
		}
	}

	if cores, ok := systemInfo["cpu_cores"].(float64); ok {
		cc.CPUInfo.Cores = int(cores)
	}
	if _, ok := systemInfo["cpu"].(float64); ok {
		cc.CPUInfo.ClockMHz = 3000
	}

	if memPercent, ok := systemInfo["memory"].(float64); ok {
		cc.VRAMGB = 16.0
		if memPercent < 30 {
			cc.VRAMGB = 8.0
		}
	}

	if totalMem, ok := systemInfo["memory_total"].(float64); ok {
		cc.VRAMGB = totalMem / (1024 * 1024 * 1024) 
	}

	cc.S_GPU = cc.calculateS_GPU()
	cc.S_CPU = cc.calculateS_CPU()
	cc.S_VRAM = cc.calculateS_VRAM()

	cudaFactor := 0.0
	if cc.CUDAEnabled {
		cudaFactor = 0.2
	}

	cc.CC = 0.6*cc.S_GPU + 0.3*cc.S_CPU + 0.1*cc.S_VRAM + cudaFactor
}

func (cc *ComputeCredit) analyzeGPU(gpuName string) GPUInfo {
	info := GPUInfo{Name: gpuName}

	if gpuName == "" || gpuName == "No GPU" {
		info.TFLOPS = 0
		info.VRAMGB = 0
		info.CUDACapable = false
		return info
	}

	lowerName := strings.ToLower(gpuName)

	if strings.Contains(lowerName, "nvidia") || strings.Contains(lowerName, "geforce") {
		info.CUDACapable = true

		switch {
		case strings.Contains(lowerName, "4090"):
			info.TFLOPS = 82.58
			info.VRAMGB = 24
		case strings.Contains(lowerName, "4080"):
			info.TFLOPS = 48.7
			info.VRAMGB = 16
		case strings.Contains(lowerName, "3090"):
			info.TFLOPS = 35.6
			info.VRAMGB = 24
		case strings.Contains(lowerName, "3080"):
			info.TFLOPS = 29.7
			info.VRAMGB = 10
		case strings.Contains(lowerName, "3070"):
			info.TFLOPS = 20.3
			info.VRAMGB = 8
		case strings.Contains(lowerName, "2080"):
			info.TFLOPS = 14.2
			info.VRAMGB = 8
		case strings.Contains(lowerName, "1080"):
			info.TFLOPS = 8.9
			info.VRAMGB = 8
		case strings.Contains(lowerName, "750"):
			info.TFLOPS = 1.4 
			info.VRAMGB = 2
		case strings.Contains(lowerName, "1060"):
			info.TFLOPS = 3.9
			info.VRAMGB = 6
		default:
			info.TFLOPS = 5.0
			info.VRAMGB = 4
		}
	} else if strings.Contains(lowerName, "amd") || strings.Contains(lowerName, "radeon") {
		info.CUDACapable = false

		switch {
		case strings.Contains(lowerName, "7900"):
			info.TFLOPS = 61.0
			info.VRAMGB = 20
		case strings.Contains(lowerName, "7800"):
			info.TFLOPS = 37.0
			info.VRAMGB = 16
		case strings.Contains(lowerName, "6700"):
			info.TFLOPS = 19.5
			info.VRAMGB = 12
		default:
			info.TFLOPS = 5.0
			info.VRAMGB = 8
		}
	} else if strings.Contains(lowerName, "intel") {
		info.TFLOPS = 0.5
		info.VRAMGB = 2 
		info.CUDACapable = false
	} else {
		info.TFLOPS = 0.5
		info.VRAMGB = 1
		info.CUDACapable = false
	}

	return info
}

func (cc *ComputeCredit) calculateS_GPU() float64 {
	return (cc.GPUInfo.TFLOPS / 100.0) + (cc.VRAMGB / 12.0)
}

func (cc *ComputeCredit) calculateS_CPU() float64 {
	cores := cc.CPUInfo.Cores
	if cores == 0 {
		cores = 4 
	}

	clock := cc.CPUInfo.ClockMHz
	if clock == 0 {
		clock = 3000
	}

	ipc := cc.CPUInfo.IPC
	if ipc == 0 {
		ipc = 1.0
	}

	cr := clock / 4000.0
	if cr > 1.0 {
		cr = 1.0
	}

	return float64(cores) * cr * ipc / 96.0
}

func (cc *ComputeCredit) calculateS_VRAM() float64 {
	vramRatio := cc.VRAMGB / 12.0
	if vramRatio > 1.0 {
		vramRatio = 1.0
	}
	return math.Min(1.0, vramRatio)
}

func GetComputeCreditForSystemInfo(systemInfo map[string]interface{}) float64 {
	cc := &ComputeCredit{}
	cc.Calculate(systemInfo)
	return cc.CC
}

func (cc *ComputeCredit) String() string {
	return fmt.Sprintf("CC=%.4f (S_gpu=%.4f, S_cpu=%.4f, S_vram=%.4f, CUDA=%v)",
		cc.CC, cc.S_GPU, cc.S_CPU, cc.S_VRAM, cc.CUDAEnabled)
}
