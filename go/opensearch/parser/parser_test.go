package parser

import (
	"reflect"
	"testing"
)

func TestBuildNestedMap(t *testing.T) {
	headerPaths := [][]string{{"level1", "level2", "level3"}}
	values := []string{"value"}

	expected := map[string]any{
		"level1": map[string]any{
			"level2": map[string]any{
				"level3": [][][]any{{{"value"}}},
			},
		},
	}

	result := buildNestedMap(headerPaths, values)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("buildNestedMap() = %+v, want %+v", result, expected)
	}
}

func TestEnsure3DArray(t *testing.T) {
	value := "1/2|3"
	expected := [][][]any{
		{
			{1, 2},
		},
		{
			{3},
		},
	}

	result := ensure3DArray(value)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ensure3DArray() = %v, want %v", result, expected)
	}
}
