package com.mina.preprocessing;

import org.apache.commons.collections4.MapUtils;

import java.util.HashMap;
import java.util.Map;

public class LabelEncoder {

    private Map<String, Integer> map = new HashMap<>();

    public int[] fitTransform(String[] values) {
        assert values != null && values.length > 0;

        int[] mappedValues = new int[values.length];
        int value = 0;
        for (int i = 0; i < values.length; i++) {
            if (!map.containsKey(values[i])) {
                mappedValues[i] = map.put(values[i], value++);
            }
        }

        return mappedValues;
    }

    public String[] inverseTransform(int[] values) {
        assert values != null && values.length > 0;

        String[] labels = new String[values.length];
        Map<Integer, String> inversedMap = MapUtils.invertMap(map);
        for (int i = 0; i < values.length; i++) {
            assert inversedMap.containsKey(values[i]);
            labels[i] = inversedMap.get(values[i]);
        }

        return labels;
    }
}
