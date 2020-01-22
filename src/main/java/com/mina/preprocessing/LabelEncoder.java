package com.mina.preprocessing;

import org.apache.commons.collections4.MapUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class LabelEncoder {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(LabelEncoder.class);

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
