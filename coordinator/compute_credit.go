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
type ComputeCredit struct {
	GPUInfo     GPUInfo `json:"gpu_info"`
	CPUInfo     CPUInfo `json:"cpu_info"`
	VRAMGB      float64 `json:"vram_gb"`
	CUDAEnabled bool    `json:"cuda_enabled"`

	// Calculated values
	S_GPU  float64 `json:"s_gpu"`
	S_CPU  float64 `json:"s_cpu"`
	S_VRAM float64 `json:"s_vram"`
	CC     float64 `json:"compute_credit"`
}

type GPUInfo struct {
	Name        string  `json:"name"`
	TFLOPS      float64 `json:"tflops"`  // FP32 TFLOPS
	VRAMGB      float64 `json:"vram_gb"` // VRAM in GB
	CUDACapable bool    `json:"cuda_capable"`
}

type CPUInfo struct {
	Model    string  `json:"model"`
	Cores    int     `json:"cores"`
	ClockMHz float64 `json:"clock_mhz"` // Base clock in MHz
	IPC      float64 `json:"ipc"`       // Instructions per cycle (relative)
}

// Calculate computes the ComputeCredit based on system info
func (cc *ComputeCredit) Calculate(systemInfo map[string]interface{}) {
	// First, check for top-level vram_gb and cuda_available from Python
	if vram, ok := systemInfo["vram_gb"].(float64); ok && vram > 0 {
		cc.GPUInfo.VRAMGB = vram
	}
	if cuda, ok := systemInfo["cuda_available"].(bool); ok {
		cc.CUDAEnabled = cuda
		cc.GPUInfo.CUDACapable = cuda
	}

	// Extract GPU info - support both old string format and new JSON format
	// Only use these if we haven't already set values from top-level fields
	if cc.GPUInfo.Name == "" && cc.GPUInfo.TFLOPS == 0 {
		if gpuObj, ok := systemInfo["gpu"].(map[string]interface{}); ok {
			// New JSON format: {"name": "...", "vram_gb": 2.0, "cuda_available": true}
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
			// GPU info is sent as JSON string - try to parse it
			var gpuMap map[string]interface{}
			if err := json.Unmarshal([]byte(gpuStr), &gpuMap); err == nil {
				// Successfully parsed JSON string
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
				// Old string format - just GPU name
				cc.GPUInfo = cc.analyzeGPU(gpuStr)
			}
		}
	}

	// Extract CPU info - properly get cores
	if cores, ok := systemInfo["cpu_cores"].(float64); ok {
		cc.CPUInfo.Cores = int(cores)
	}

	// Get CPU usage percentage (for clock estimation)
	if _, ok := systemInfo["cpu"].(float64); ok {
		// Estimate clock from usage - higher usage might mean more activity
		// Base on typical 3GHz CPU
		cc.CPUInfo.ClockMHz = 3000
	}

	// Extract actual memory in GB (not percentage)
	// Python sends memory as percentage, but we need to estimate total
	if memPercent, ok := systemInfo["memory"].(float64); ok {
		// Estimate based on typical system: if 50% used of 16GB, use 16GB
		// For now, assume 16GB base for systems with memory reporting
		cc.VRAMGB = 16.0 // Base assumption for integrated GPU
		// If percentage is very low, might indicate smaller system
		if memPercent < 30 {
			cc.VRAMGB = 8.0
		}
	}

	// If we have actual memory info, use it
	if totalMem, ok := systemInfo["memory_total"].(float64); ok {
		cc.VRAMGB = totalMem / (1024 * 1024 * 1024) // Convert bytes to GB
	}

	// Calculate components
	cc.S_GPU = cc.calculateS_GPU()
	cc.S_CPU = cc.calculateS_CPU()
	cc.S_VRAM = cc.calculateS_VRAM()

	// Calculate final ComputeCredit
	// CC = 0.6*S(gpu) + 0.3*S(cpu) + 0.1*S(vram) + 0.2*CUDA_ENABLED
	cudaFactor := 0.0
	if cc.CUDAEnabled {
		cudaFactor = 0.2
	}

	cc.CC = 0.6*cc.S_GPU + 0.3*cc.S_CPU + 0.1*cc.S_VRAM + cudaFactor
}

func (cc *ComputeCredit) analyzeGPU(gpuName string) GPUInfo {
	info := GPUInfo{Name: gpuName}

	// Handle empty or "No GPU" case
	if gpuName == "" || gpuName == "No GPU" {
		info.TFLOPS = 0
		info.VRAMGB = 0
		info.CUDACapable = false
		return info
	}

	// Analyze GPU name and estimate specs
	lowerName := strings.ToLower(gpuName)

	// Check for NVIDIA GPUs and estimate TFLOPS
	if strings.Contains(lowerName, "nvidia") || strings.Contains(lowerName, "geforce") {
		info.CUDACapable = true

		// Estimate based on model series
		switch {
		case strings.Contains(lowerName, "4090"):
			info.TFLOPS = 82.58 // FP32 TFLOPS for RTX 4090
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
			info.TFLOPS = 1.4 // FP32 TFLOPS for GTX 750 Ti
			info.VRAMGB = 2
		case strings.Contains(lowerName, "1060"):
			info.TFLOPS = 3.9
			info.VRAMGB = 6
		default:
			// Generic NVIDIA GPU estimate
			info.TFLOPS = 5.0
			info.VRAMGB = 4
		}
	} else if strings.Contains(lowerName, "amd") || strings.Contains(lowerName, "radeon") {
		info.CUDACapable = false

		// Estimate AMD GPU specs
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
		// Integrated graphics - minimal compute
		info.TFLOPS = 0.5
		info.VRAMGB = 2 // Uses system RAM
		info.CUDACapable = false
	} else {
		// Generic/unknown GPU - assume minimal
		info.TFLOPS = 0.5
		info.VRAMGB = 1
		info.CUDACapable = false
	}

	return info
}

func (cc *ComputeCredit) calculateS_GPU() float64 {
	// S(gpu) = FP(x)TFLOPS/100 + VRAM/12
	return (cc.GPUInfo.TFLOPS / 100.0) + (cc.VRAMGB / 12.0)
}

func (cc *ComputeCredit) calculateS_CPU() float64 {
	// S(cpu) = Cores_number * CR * IPCrel/96
	// If we don't have exact values, use defaults
	cores := cc.CPUInfo.Cores
	if cores == 0 {
		cores = 4 // Default assumption
	}

	clock := cc.CPUInfo.ClockMHz
	if clock == 0 {
		clock = 3000 // Default 3GHz
	}

	ipc := cc.CPUInfo.IPC
	if ipc == 0 {
		ipc = 1.0 // Relative IPC baseline
	}

	// Calculate relative clock rate (assuming 4GHz = 1.0)
	cr := clock / 4000.0
	if cr > 1.0 {
		cr = 1.0
	}

	return float64(cores) * cr * ipc / 96.0
}

func (cc *ComputeCredit) calculateS_VRAM() float64 {
	// S(vram) = min(1, VRAM/12)
	vramRatio := cc.VRAMGB / 12.0
	if vramRatio > 1.0 {
		vramRatio = 1.0
	}
	return math.Min(1.0, vramRatio)
}

// GetComputeCreditForSystemInfo calculates CC from raw system info map
func GetComputeCreditForSystemInfo(systemInfo map[string]interface{}) float64 {
	cc := &ComputeCredit{}
	cc.Calculate(systemInfo)
	return cc.CC
}

// String returns a formatted string with ComputeCredit breakdown
func (cc *ComputeCredit) String() string {
	return fmt.Sprintf("CC=%.4f (S_gpu=%.4f, S_cpu=%.4f, S_vram=%.4f, CUDA=%v)",
		cc.CC, cc.S_GPU, cc.S_CPU, cc.S_VRAM, cc.CUDAEnabled)
}
