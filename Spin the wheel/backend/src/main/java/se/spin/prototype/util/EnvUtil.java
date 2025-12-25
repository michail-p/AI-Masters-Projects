package se.spin.prototype.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public final class EnvUtil {

    private static final Logger log = LoggerFactory.getLogger(EnvUtil.class);
    private static final String DOTENV = ".env";
    private static Map<String, String> cache;

    private EnvUtil() {}


    public static String get(String key) {

        String val = System.getenv(key);

        if (val != null && !val.isBlank()) 
            return val;

        ensureLoaded();

        return cache.get(key);
    }

    private static synchronized void ensureLoaded() {

        if (cache != null) 
            return;

        cache = new HashMap<>();
        Path path = Path.of(System.getProperty("user.dir"), DOTENV);

        if (!Files.exists(path)) 
            return;

        try {

            Files.lines(path)
                .map(String::trim)
                .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                .forEach(line -> {

                    int idx = line.indexOf('=');

                    if (idx > 0 && idx < line.length() - 1) {
                        String k = line.substring(0, idx).trim();
                        String v = line.substring(idx + 1).trim();
                        cache.put(k, v);
                    }

                });

        } catch (IOException e) {
            log.warn("Failed to read .env file", e);
        }
    }
}