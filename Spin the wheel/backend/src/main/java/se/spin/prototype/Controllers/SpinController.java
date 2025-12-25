package se.spin.prototype.Controllers;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import org.springframework.web.server.ResponseStatusException;
import se.spin.prototype.services.FirestoreService;
import se.spin.prototype.services.HuggingFaceService;
import se.spin.prototype.Beans.CompareScenariosRequest;
import se.spin.prototype.Beans.GeneratedTextSources;
import se.spin.prototype.Beans.SeedResult;
import se.spin.prototype.Beans.SpinArguments;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/spin")
public class SpinController {

    private final FirestoreService firestoreService;
    private final HuggingFaceService huggingFaceService;

    public SpinController(FirestoreService firestoreService, HuggingFaceService huggingFaceService) {
        this.firestoreService = firestoreService;
        this.huggingFaceService = huggingFaceService;
    }

    @PostMapping(value = "/story/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter postGeneratedTextStream(@RequestBody SpinArguments arguments) {
        validateSpinArguments(arguments);

        SeedResult seedResult = firestoreService.fetchSeedText(
            arguments.getCity(),
            arguments.getYear(),
            arguments.getGender()
        ).orElse(new SeedResult("No matching Firestore seed; use the provided context to craft a new story.", null));

        SseEmitter emitter = new SseEmitter(0L);

        // Send sources metadata first
        try {
            List<String> sources = buildSources(seedResult.getLink());
            emitter.send(SseEmitter.event().name("sources").data(String.join(",", sources)));
        } catch (IOException e) {
            emitter.completeWithError(e);
            return emitter;
        }

        huggingFaceService.streamStory(arguments, seedResult.getText())
            .doOnNext(chunk -> {
                try {
                    emitter.send(SseEmitter.event().data(chunk));
                } catch (IOException e) {
                    emitter.completeWithError(e);
                }
            })
            .doOnComplete(emitter::complete)
            .doOnError(emitter::completeWithError)
            .subscribe();

        return emitter;
    }

    @PostMapping("/image")
    public ResponseEntity<ByteArrayResource> postGeneratedImage(@RequestBody SpinArguments body) {

        validateSpinArguments(body);

        SeedResult seedResult = firestoreService.fetchSeedText(
            body.getCity(),
            body.getYear(),
            body.getGender()
        ).orElse(new SeedResult("No matching Firestore seed; use the provided context to craft a new story.", null));

        var imageResult = huggingFaceService.generateImage(body, seedResult.getText());

        return ResponseEntity.ok()
                .contentType(imageResult.contentType())
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"story-image.png\"")
                .body(new ByteArrayResource(imageResult.data()));
    }

    @PostMapping(value = "/compare-scenarios/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter postCompareScenariosStream(@RequestBody CompareScenariosRequest arguments) {

        if (arguments == null || arguments.getSpinArgumentsFirstStory() == null || arguments.getSpinArgumentsSecondStory() == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Both scenarios are required");
        }

        validateSpinArguments(arguments.getSpinArgumentsFirstStory());
        validateSpinArguments(arguments.getSpinArgumentsSecondStory());

        SpinArguments firstArgs = arguments.getSpinArgumentsFirstStory();
        SpinArguments secondArgs = arguments.getSpinArgumentsSecondStory();

        SeedResult firstSeed = firestoreService.fetchSeedText(
                firstArgs.getCity(),
                firstArgs.getYear(),
                firstArgs.getGender()
            ).orElse(new SeedResult("No matching Firestore seed; use the provided context to craft a new story.", null));
        
        SeedResult secondSeed = firestoreService.fetchSeedText(
                secondArgs.getCity(),
                secondArgs.getYear(),
                secondArgs.getGender()
            ).orElse(new SeedResult("No matching Firestore seed; use the provided context to craft a new story.", null));

        String firstStory = huggingFaceService.generateStory(firstArgs, firstSeed.getText());
        String secondStory = huggingFaceService.generateStory(secondArgs, secondSeed.getText());

        SseEmitter emitter = new SseEmitter(0L);

        // Send sources metadata first
        try {
            List<String> sources = new ArrayList<>();
            sources.addAll(buildSources(firstSeed.getLink()));
            sources.addAll(buildSources(secondSeed.getLink()));
            emitter.send(SseEmitter.event().name("sources").data(String.join(",", sources)));
        } catch (IOException e) {
            emitter.completeWithError(e);
            return emitter;
        }

        huggingFaceService.streamCompareStories(firstArgs, secondArgs, firstStory, secondStory)
            .doOnNext(chunk -> {
                try {
                    emitter.send(SseEmitter.event().data(chunk));
                } catch (IOException e) {
                    emitter.completeWithError(e);
                }
            })
            .doOnComplete(emitter::complete)
            .doOnError(emitter::completeWithError)
            .subscribe();

        return emitter;
    }

    private void validateSpinArguments(SpinArguments arguments) {
        if (arguments == null || arguments.getCity() == null || arguments.getYear() == null || arguments.getGender() == null || arguments.getGender().getId() == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "City, year and gender are required");
        }
    }

    private List<String> buildSources(String skblLink) {
        List<String> sources = new ArrayList<>();
        if (skblLink != null && !skblLink.isBlank()) {
            sources.add(skblLink);
        }
        return sources;
    }
}
