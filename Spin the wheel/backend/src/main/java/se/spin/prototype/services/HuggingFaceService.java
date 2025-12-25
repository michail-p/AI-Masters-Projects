package se.spin.prototype.services;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Flux;
import se.spin.prototype.Beans.SpinArguments;
import se.spin.prototype.util.EnvUtil;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.web.util.UriComponentsBuilder;

@Component
public class HuggingFaceService {

    private static final Logger log = LoggerFactory.getLogger(HuggingFaceService.class);

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final WebClient webClient = WebClient.builder().build();


    private String buildStoryPrompt(SpinArguments arguments, String seedText) {

        StringBuilder sb = new StringBuilder();

        sb.append("Write a short, realistic story set in ").append(arguments.getCity())
            .append(" around year ").append(arguments.getYear())
            .append(" about a ").append(arguments.getGender().getDescription())
            .append("\n\nContext: ").append(seedText)
            .append("\n\nReturn only the story text.");

        return sb.toString();
    }

    private String buildComparePrompt(SpinArguments firstArgs, SpinArguments secondArgs, String firstStory, String secondStory) {

        return "Compare the two historical stories below. Highlight key differences in setting, tone, and perspective. Be concise (max 6 sentences). " +
            "Story 1 (" + firstArgs.getCity() + ", " + firstArgs.getYear() + ", " + firstArgs.getGender().getDescription() + "):\n" + firstStory + "\n\n" +
            "Story 2 (" + secondArgs.getCity() + ", " + secondArgs.getYear() + ", " + secondArgs.getGender().getDescription() + "):\n" + secondStory;

    }

    public String generateStory(SpinArguments arguments, String seedText) {

        String prompt = buildStoryPrompt(arguments, seedText);

        Map<String, Object> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", prompt);

        Map<String, Object> payload = new HashMap<>();
        payload.put("model", "AI-Sweden-Models/Llama-3-8B-instruct:featherless-ai");
        payload.put("messages", List.of(userMessage));
        payload.put("max_tokens", 256);
        payload.put("temperature", 0.7);
        payload.put("top_p", 0.9);
        payload.put("stream", false);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setAccept(List.of(MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN));
        headers.setBearerAuth(EnvUtil.get("HUGGINGFACE_API_TOKEN"));
        headers.add("X-Router-Provider", "nscale");
        
        try {

            ResponseEntity<String> response = restTemplate.exchange(
                "https://router.huggingface.co/v1/chat/completions",
                HttpMethod.POST, 
                new HttpEntity<>(payload, headers), 
                String.class
            );

            if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
                log.error("HuggingFace call failed: status {} body {}", response.getStatusCode(), response.getBody());
                throw new ResponseStatusException(response.getStatusCode(), "HuggingFace generation failed");
            }
            
            String result = extractText(response.getBody());

            return result;

        } catch (org.springframework.web.client.HttpStatusCodeException ex) {

            String body = ex.getResponseBodyAsString();
            log.error("HuggingFace text error: status {} body {}", ex.getStatusCode(), body);
            throw new ResponseStatusException(ex.getStatusCode(), "HuggingFace text error: " + body);

        }
    }

    public Flux<String> streamStory(SpinArguments arguments, String seedText) {

        String prompt = buildStoryPrompt(arguments, seedText);

        Map<String, Object> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", prompt);

        Map<String, Object> payload = new HashMap<>();
        payload.put("model", "AI-Sweden-Models/Llama-3-8B-instruct:featherless-ai");
        payload.put("messages", List.of(userMessage));
        payload.put("max_tokens", 256);
        payload.put("temperature", 0.7);
        payload.put("top_p", 0.9);
        payload.put("stream", true);

        return webClient.post()
            .uri("https://router.huggingface.co/v1/chat/completions")
            .headers(h -> {
                h.setContentType(MediaType.APPLICATION_JSON);
                h.setAccept(List.of(MediaType.TEXT_EVENT_STREAM, MediaType.ALL));
                h.setBearerAuth(EnvUtil.get("HUGGINGFACE_API_TOKEN"));
                h.add("X-Router-Provider", "nscale");
            })
            .bodyValue(payload)
            .retrieve()
            .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {})
            .flatMap(sse -> {
                String data = sse.data();
                if (data == null || data.isBlank()) return Flux.empty();
                return Flux.fromIterable(extractStreamDelta(data));
            })
            .onErrorResume(ex -> {
                log.error("HuggingFace streaming error", ex);
                return Flux.error(new ResponseStatusException(org.springframework.http.HttpStatus.INTERNAL_SERVER_ERROR, "HuggingFace streaming failed", ex));
            });
    }

    public String compareStories(SpinArguments firstArgs, SpinArguments secondArgs, String firstStory, String secondStory) {

        String prompt = buildComparePrompt(firstArgs, secondArgs, firstStory, secondStory);

        Map<String, Object> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", prompt);

        return sendChatCompletion(userMessage, 256);
    }

    public Flux<String> streamCompareStories(SpinArguments firstArgs, SpinArguments secondArgs, String firstStory, String secondStory) {

        String prompt = buildComparePrompt(firstArgs, secondArgs, firstStory, secondStory);

        Map<String, Object> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", prompt);

        Map<String, Object> payload = new HashMap<>();
        payload.put("model", "AI-Sweden-Models/Llama-3-8B-instruct:featherless-ai");
        payload.put("messages", List.of(userMessage));
        payload.put("max_tokens", 256);
        payload.put("temperature", 0.7);
        payload.put("top_p", 0.9);
        payload.put("stream", true);

        return webClient.post()
            .uri("https://router.huggingface.co/v1/chat/completions")
            .headers(h -> {
                h.setContentType(MediaType.APPLICATION_JSON);
                h.setAccept(List.of(MediaType.TEXT_EVENT_STREAM, MediaType.ALL));
                h.setBearerAuth(EnvUtil.get("HUGGINGFACE_API_TOKEN"));
                h.add("X-Router-Provider", "nscale");
            })
            .bodyValue(payload)
            .retrieve()
            .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {})
            .flatMap(sse -> {
                String data = sse.data();
                if (data == null || data.isBlank()) return Flux.empty();
                return Flux.fromIterable(extractStreamDelta(data));
            })
            .onErrorResume(ex -> {
                log.error("HuggingFace streaming compare error", ex);
                return Flux.error(new ResponseStatusException(org.springframework.http.HttpStatus.INTERNAL_SERVER_ERROR, "HuggingFace streaming compare failed", ex));
            });
    }

    private String sendChatCompletion(Map<String, Object> userMessage, int maxTokens) {

        Map<String, Object> payload = new HashMap<>();
        payload.put("model", "AI-Sweden-Models/Llama-3-8B-instruct:featherless-ai");
        payload.put("messages", List.of(userMessage));
        payload.put("max_tokens", maxTokens);
        payload.put("temperature", 0.7);
        payload.put("top_p", 0.9);
        payload.put("stream", false);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setAccept(List.of(MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN));
        headers.setBearerAuth(EnvUtil.get("HUGGINGFACE_API_TOKEN"));
        headers.add("X-Router-Provider", "nscale");
        
        try {

            ResponseEntity<String> response = restTemplate.exchange(
                "https://router.huggingface.co/v1/chat/completions",
                HttpMethod.POST, 
                new HttpEntity<>(payload, headers), 
                String.class
            );

            if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
                log.error("HuggingFace call failed: status {} body {}", response.getStatusCode(), response.getBody());
                throw new ResponseStatusException(response.getStatusCode(), "HuggingFace generation failed");
            }
            
            return extractText(response.getBody());

        } catch (org.springframework.web.client.HttpStatusCodeException ex) {

            String body = ex.getResponseBodyAsString();
            log.error("HuggingFace text error: status {} body {}", ex.getStatusCode(), body);
            throw new ResponseStatusException(ex.getStatusCode(), "HuggingFace text error: " + body);

        }
    }

    private Iterable<String> extractStreamDelta(String chunk) {
        // Each SSE event comes as lines starting with "data: {...}". Ignore control messages.
        try {
            String dataLine = chunk.startsWith("data:") ? chunk.substring(5).trim() : chunk;
            if ("[DONE]".equals(dataLine)) {
                return List.of();
            }

            JsonNode node = objectMapper.readTree(dataLine);
            if (node.has("choices") && node.get("choices").isArray() && node.get("choices").size() > 0) {
                JsonNode delta = node.get("choices").get(0).path("delta");

                if (delta.has("content")) {
                    JsonNode contentNode = delta.get("content");

                    if (contentNode.isTextual()) {
                        return List.of(contentNode.asText());
                    }

                    if (contentNode.isArray()) {
                        StringBuilder sb = new StringBuilder();
                        for (JsonNode n : contentNode) {
                            if (n.has("text")) {
                                sb.append(n.get("text").asText());
                            }
                        }
                        if (sb.length() > 0) return List.of(sb.toString());
                    }
                }
            }

            return List.of();

        } catch (Exception ex) {
            log.warn("Failed to parse stream chunk: {}", chunk, ex);
            return List.of();
        }
    }

    public ImageResult generateImage(SpinArguments arguments, String seedText) {

        int width = 512;
        int height = 512;

        String prompt = buildStoryPrompt(arguments, seedText);

        URI uri = UriComponentsBuilder
            .fromHttpUrl("https://image.pollinations.ai/prompt/{prompt}")
            .queryParam("width", width)
            .queryParam("height", height)
            .buildAndExpand(prompt)
            .encode()
            .toUri();

        HttpHeaders headers = new HttpHeaders();
        headers.setAccept(List.of(MediaType.IMAGE_PNG, MediaType.IMAGE_JPEG, MediaType.ALL));

        try {

            ResponseEntity<byte[]> response = restTemplate.exchange(
                uri,
                HttpMethod.GET,
                new HttpEntity<>(headers),
                byte[].class
            );

            if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null || response.getBody().length == 0) {
                log.error("Pollinations image call failed: status {}", response.getStatusCode());
                throw new ResponseStatusException(org.springframework.http.HttpStatus.BAD_GATEWAY,
                    "Pollinations image generation failed");
            }

            MediaType contentType = response.getHeaders().getContentType();
            if (contentType == null) {
                contentType = MediaType.IMAGE_JPEG;
            }

            return new ImageResult(response.getBody(), contentType);

        } catch (org.springframework.web.client.HttpStatusCodeException ex) {
            String body = ex.getResponseBodyAsString();
            log.error("Pollinations image error: status {} body {}", ex.getStatusCode(), body);
            throw new ResponseStatusException(ex.getStatusCode(), "Pollinations image error: " + body);
        } catch (Exception ex) {
            log.error("Pollinations image request failed", ex);
            throw new ResponseStatusException(org.springframework.http.HttpStatus.BAD_GATEWAY,
                "Pollinations image generation failed", ex);
        }
    }

    private String extractText(String body) {

        try {

            JsonNode node = objectMapper.readTree(body);

            // OpenAI-compatible chat completion: choices[0].message.content
            if (node.has("choices") && node.get("choices").isArray() && node.get("choices").size() > 0) {
                JsonNode firstChoice = node.get("choices").get(0);
                if (firstChoice.has("message") && firstChoice.get("message").has("content")) {
                    return firstChoice.get("message").get("content").asText();
                }
            }

            // Legacy inference output: [{"generated_text": "..."}] or {"generated_text": "..."}
            if (node.isArray() && node.size() > 0 && node.get(0).has("generated_text")) {
                return node.get(0).get("generated_text").asText();
            }
            if (node.isObject() && node.has("generated_text")) {
                return node.get("generated_text").asText();
            }

            // Fallback: raw text
            if (node.isTextual()) {
                return node.asText();
            }

            log.error("Unexpected HuggingFace response: {}", body);

            throw new ResponseStatusException(org.springframework.http.HttpStatus.INTERNAL_SERVER_ERROR, 
                "Unexpected HuggingFace response format");

        } catch (IOException e) {
            throw new ResponseStatusException(org.springframework.http.HttpStatus.INTERNAL_SERVER_ERROR, 
                "Failed to parse HuggingFace response", e);
        }
    }

    public record ImageResult(byte[] data, MediaType contentType) {}
}