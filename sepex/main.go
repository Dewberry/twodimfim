package main

import (
	"encoding/json"
	"flag"
	"io"
	"log"
	"os"
	"path/filepath"
)

func parseInput(data any) any {
	switch v := data.(type) {
	case string:
		var parsed any
		if err := json.Unmarshal([]byte(v), &parsed); err == nil {
			return parsed
		}
		return v
	case map[string]any:
		return v
	default:
		return data
	}
}

func main() {
	printInput := flag.Bool("print-input", false, "Print the parsed input to stdout")
	flag.Parse()

	stdinData, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("Error reading from stdin: %v", err)
	}

	var payload any
	err = json.Unmarshal(stdinData, &payload)
	if err != nil {
		payload = string(stdinData)
	}

	payload = parseInput(payload)

	outputDir := "/mnt"
	err = os.MkdirAll(outputDir, 0755)
	if err != nil {
		log.Fatalf("Error creating directory %s: %v", outputDir, err)
	}

	outputPath := filepath.Join(outputDir, "payload")

	outputData, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		log.Fatalf("Error marshaling JSON: %v", err)
	}

	err = os.WriteFile(outputPath, outputData, 0644)
	if err != nil {
		log.Fatalf("Error writing to file %s: %v", outputPath, err)
	}

	log.Printf("Successfully wrote payload to %s\n", outputPath)
	if *printInput {
		log.Printf("Input: %v\n", payload)
	}
}
