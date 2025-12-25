package se.spin.prototype.services;

import com.google.api.core.ApiFuture;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.Query;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.QuerySnapshot;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.cloud.FirestoreClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;
import se.spin.prototype.Beans.Gender;
import se.spin.prototype.Beans.GenderEnum;
import se.spin.prototype.Beans.SeedResult;
import se.spin.prototype.util.EnvUtil;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

@Component
public class FirestoreService {

    private static final Logger log = LoggerFactory.getLogger(FirestoreService.class);
    private final Firestore firestore;
    private final boolean firestoreEnabled;

    public FirestoreService() {
        this.firestore = initFirestoreOrNull();
        this.firestoreEnabled = (this.firestore != null);
        //seedStories();
    }


    public Optional<SeedResult> fetchSeedText(String city, Integer year, Gender gender) {

        if (!firestoreEnabled) {
            return Optional.empty();
        }

        if (gender == null || gender.getId() == null) {
            return Optional.empty();
        }

        try {

            Query query = firestore.collection("stories")
                .whereEqualTo("city", city)
                .whereEqualTo("year", year)
                .whereEqualTo("gender", gender.getId().name())
                .limit(1);

            QuerySnapshot snapshot = query.get().get();
            if (snapshot.isEmpty()) {
                return Optional.empty();
            }
            
            var doc = snapshot.getDocuments().get(0);
            Object text = doc.get("text");
            Object link = doc.get("link");

            if (text == null) {
                return Optional.empty();
            }
            
            return Optional.of(new SeedResult(
                text.toString(),
                link != null ? link.toString() : null
            ));

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Interrupted while querying Firestore", e);
        } catch (ExecutionException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to query Firestore", e);
        }
    }

    private Firestore initFirestoreOrNull() {

        try {

            String credentialsPath = EnvUtil.get("GOOGLE_APPLICATION_CREDENTIALS");
            String projectId = EnvUtil.get("FIREBASE_PROJECT_ID");

            if (credentialsPath == null || credentialsPath.isBlank() || projectId == null || projectId.isBlank()) {
                log.warn("Firestore disabled (missing GOOGLE_APPLICATION_CREDENTIALS and/or FIREBASE_PROJECT_ID)");
                return null;
            }

            if (FirebaseApp.getApps().isEmpty()) {

                FirebaseOptions.Builder builder = FirebaseOptions.builder();
                builder.setCredentials(GoogleCredentials.fromStream(new FileInputStream(credentialsPath)));
                builder.setProjectId(projectId);

                FirebaseApp.initializeApp(builder.build());
            }
            
            return FirestoreClient.getFirestore();

        } catch (IOException e) {
            log.error("Failed to initialize Firestore; running with Firestore disabled", e);
            return null;
        }
    }

    /**
    * Seed the "stories" collection with predefined story documents.
    *
    * Each document will have the fields:
    *  - city   (String)
    *  - year   (Integer)
    *  - gender (String, matching GenderEnum.name())
    *  - text   (String)
    */
    public void seedStories() {

        List<StorySeed> seeds = Arrays.asList(
            // Stockholm – Male
            new StorySeed("Stockholm", 1800, GenderEnum.MALE,
                "Born in 1800 in Stockholm, he grew up amid cobblestone streets and merchant ships, learning a traditional craft from his father.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1780&born_end=1820"),
            new StorySeed("Stockholm", 1850, GenderEnum.MALE,
                "Born in 1850 in Stockholm, he witnessed the city's early industrial growth and spent his youth working in emerging factories along the waterways.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1830&born_end=1870"),
            new StorySeed("Stockholm", 1900, GenderEnum.MALE,
                "Born in 1900 in Stockholm, he came of age between two world wars, navigating rapid social change and the rise of modern Swedish democracy.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1880&born_end=1920"),
            new StorySeed("Stockholm", 1950, GenderEnum.MALE,
                "Born in 1950 in Stockholm, he enjoyed the postwar welfare boom, studying at university and later working in a growing service economy.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1930&born_end=1970"),
            new StorySeed("Stockholm", 2000, GenderEnum.MALE,
                "Born in 2000 in Stockholm, he is a digitally native urban professional who balances startup work with fika in trendy Södermalm cafés.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1980&born_end=2020"),

            // Stockholm – Female
            new StorySeed("Stockholm", 1800, GenderEnum.FEMALE,
                "Born in 1800 in Stockholm, she lived a modest life helping her family with household work and local trade near the old harbor.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1780&born_end=1820"),
            new StorySeed("Stockholm", 1850, GenderEnum.FEMALE,
                "Born in 1850 in Stockholm, she saw railways and factories transform the city while she contributed to her household through seamstress work.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1830&born_end=1870"),
            new StorySeed("Stockholm", 1900, GenderEnum.FEMALE,
                "Born in 1900 in Stockholm, she lived through suffrage reforms and the world wars, eventually joining the urban workforce in offices and shops.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1880&born_end=1920"),
            new StorySeed("Stockholm", 1950, GenderEnum.FEMALE,
                "Born in 1950 in Stockholm, she benefited from expanding education and social programs, building a career in public service.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1930&born_end=1970"),
            new StorySeed("Stockholm", 2000, GenderEnum.FEMALE,
                "Born in 2000 in Stockholm, she is a globally connected creative who splits her time between tech, culture, and travel.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1980&born_end=2020"),

            // Stockholm – Non-binary
            new StorySeed("Stockholm", 1800, GenderEnum.NONBINARY,
                "Born in 1800 in Stockholm, they navigated a rigid society while quietly carving out a life among artisans and traders near the old harbor.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1780&born_end=1820"),
            new StorySeed("Stockholm", 1850, GenderEnum.NONBINARY,
                "Born in 1850 in Stockholm, they found solace in the anonymity of industrial growth, working in factories where skill mattered more than conformity.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1830&born_end=1870"),
            new StorySeed("Stockholm", 1900, GenderEnum.NONBINARY,
                "Born in 1900 in Stockholm, they witnessed social upheaval and quietly advocated for broader acceptance while working in the city's growing bureaucracy.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1880&born_end=1920"),
            new StorySeed("Stockholm", 1950, GenderEnum.NONBINARY,
                "Born in 1950 in Stockholm, they benefited from Sweden's progressive social reforms and found community among artists and intellectuals.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1930&born_end=1970"),
            new StorySeed("Stockholm", 2000, GenderEnum.NONBINARY,
                "Born in 2000 in Stockholm, they are a digital creative thriving in one of Europe's most inclusive cities, blending tech work with LGBTQ+ advocacy.",
                "https://skbl.se/sv/artikel/search?location=Stockholm&born_start=1980&born_end=2020"),

            // Göteborg – Male
            new StorySeed("Göteborg", 1800, GenderEnum.MALE,
                "Born in 1800 in Göteborg, he worked along the busy port, hauling goods and learning seafaring skills from older sailors.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1780&born_end=1820"),
            new StorySeed("Göteborg", 1850, GenderEnum.MALE,
                "Born in 1850 in Göteborg, he joined the shipyards as a young man, shaping iron and wood for trading vessels bound across the North Sea.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1830&born_end=1870"),
            new StorySeed("Göteborg", 1900, GenderEnum.MALE,
                "Born in 1900 in Göteborg, he labored in heavy industry and shipbuilding, raising a family in a working-class district near the docks.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1880&born_end=1920"),
            new StorySeed("Göteborg", 1950, GenderEnum.MALE,
                "Born in 1950 in Göteborg, he became part of the automotive era, finding stable employment at a major car factory.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1930&born_end=1970"),
            new StorySeed("Göteborg", 2000, GenderEnum.MALE,
                "Born in 2000 in Göteborg, he is a music-loving engineer who bikes to work and spends weekends at coastal islands nearby.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1980&born_end=2020"),

            // Göteborg – Female
            new StorySeed("Göteborg", 1800, GenderEnum.FEMALE,
                "Born in 1800 in Göteborg, she kept a small home near the harbor, supporting traders and sailors through cooking and mending clothes.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1780&born_end=1820"),
            new StorySeed("Göteborg", 1850, GenderEnum.FEMALE,
                "Born in 1850 in Göteborg, she helped run a family shop supplying goods to shipyard workers while raising several children.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1830&born_end=1870"),
            new StorySeed("Göteborg", 1900, GenderEnum.FEMALE,
                "Born in 1900 in Göteborg, she saw the heyday of shipbuilding and later joined the workforce as a clerk in a bustling port office.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1880&born_end=1920"),
            new StorySeed("Göteborg", 1950, GenderEnum.FEMALE,
                "Born in 1950 in Göteborg, she studied in newly expanded schools and later worked in healthcare for industrial families.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1930&born_end=1970"),
            new StorySeed("Göteborg", 2000, GenderEnum.FEMALE,
                "Born in 2000 in Göteborg, she is a university-educated designer blending Scandinavian aesthetics with sustainable urban living.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1980&born_end=2020"),

            // Göteborg – Non-binary
            new StorySeed("Göteborg", 1800, GenderEnum.NONBINARY,
                "Born in 1800 in Göteborg, they lived discreetly among dock workers and merchants, finding acceptance in the transient world of maritime trade.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1780&born_end=1820"),
            new StorySeed("Göteborg", 1850, GenderEnum.NONBINARY,
                "Born in 1850 in Göteborg, they worked in the shipyards where hard labor overshadowed questions of identity, building a quiet life by the sea.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1830&born_end=1870"),
            new StorySeed("Göteborg", 1900, GenderEnum.NONBINARY,
                "Born in 1900 in Göteborg, they navigated industrial society while forming close bonds with fellow workers in the labor movement.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1880&born_end=1920"),
            new StorySeed("Göteborg", 1950, GenderEnum.NONBINARY,
                "Born in 1950 in Göteborg, they found community in the city's growing cultural scene and worked in education during Sweden's reform era.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1930&born_end=1970"),
            new StorySeed("Göteborg", 2000, GenderEnum.NONBINARY,
                "Born in 2000 in Göteborg, they are a sustainability-focused engineer who celebrates the city's inclusive music and arts festivals.",
                "https://skbl.se/sv/artikel/search?location=G%C3%B6teborg&born_start=1980&born_end=2020"),

            // Malmö – Male
            new StorySeed("Malmö", 1800, GenderEnum.MALE,
                "Born in 1800 in Malmö, he tilled nearby fields and brought produce into the small coastal town's market square.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1780&born_end=1820"),
            new StorySeed("Malmö", 1850, GenderEnum.MALE,
                "Born in 1850 in Malmö, he shifted from farming to factory work as the town industrialized and rail connections expanded.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1830&born_end=1870"),
            new StorySeed("Malmö", 1900, GenderEnum.MALE,
                "Born in 1900 in Malmö, he endured economic swings and war years while working in manufacturing and living in dense worker housing.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1880&born_end=1920"),
            new StorySeed("Malmö", 1950, GenderEnum.MALE,
                "Born in 1950 in Malmö, he watched the city's migrant communities grow and eventually worked on infrastructure that modernized the region.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1930&born_end=1970"),
            new StorySeed("Malmö", 2000, GenderEnum.MALE,
                "Born in 2000 in Malmö, he is a multicultural gamer and entrepreneur who commutes over the Öresund Bridge for work and leisure.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1980&born_end=2020"),

            // Malmö – Female
            new StorySeed("Malmö", 1800, GenderEnum.FEMALE,
                "Born in 1800 in Malmö, she lived a rural-urban life, spinning wool and tending animals while selling goods at the town market.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1780&born_end=1820"),
            new StorySeed("Malmö", 1850, GenderEnum.FEMALE,
                "Born in 1850 in Malmö, she supported her family by taking in laundry and later by working in a textile mill.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1830&born_end=1870"),
            new StorySeed("Malmö", 1900, GenderEnum.FEMALE,
                "Born in 1900 in Malmö, she experienced both hardship and growing rights for women, eventually working outside the home in local industry.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1880&born_end=1920"),
            new StorySeed("Malmö", 1950, GenderEnum.FEMALE,
                "Born in 1950 in Malmö, she raised children in a new housing estate while working part-time in retail in the city center.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1930&born_end=1970"),
            new StorySeed("Malmö", 2000, GenderEnum.FEMALE,
                "Born in 2000 in Malmö, she is a cosmopolitan student and activist engaged in arts, climate issues, and cross-border culture.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1980&born_end=2020"),

            // Malmö – Non-binary
            new StorySeed("Malmö", 1800, GenderEnum.NONBINARY,
                "Born in 1800 in Malmö, they lived on the margins of the small coastal town, finding work and community among traveling merchants.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1780&born_end=1820"),
            new StorySeed("Malmö", 1850, GenderEnum.NONBINARY,
                "Born in 1850 in Malmö, they adapted to industrialization by learning trades that valued skill over social conformity.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1830&born_end=1870"),
            new StorySeed("Malmö", 1900, GenderEnum.NONBINARY,
                "Born in 1900 in Malmö, they quietly built a life through manufacturing work while dreaming of a more accepting future.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1880&born_end=1920"),
            new StorySeed("Malmö", 1950, GenderEnum.NONBINARY,
                "Born in 1950 in Malmö, they found belonging in the city's diverse immigrant communities and worked in social services.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1930&born_end=1970"),
            new StorySeed("Malmö", 2000, GenderEnum.NONBINARY,
                "Born in 2000 in Malmö, they are a boundary-crossing creative who embraces the city's multicultural identity and commutes to Copenhagen for inspiration.",
                "https://skbl.se/sv/artikel/search?location=Malm%C3%B6&born_start=1980&born_end=2020")
        );

        try {
            for (StorySeed seed : seeds) {
            Map<String, Object> doc = new HashMap<>();
            doc.put("city", seed.getCity());
            doc.put("year", seed.getYear());
            doc.put("gender", seed.getGender().name());
            doc.put("text", seed.getText());
            doc.put("link", seed.getLink());

            firestore.collection("stories").add(doc).get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Interrupted while seeding Firestore stories", e);
        } catch (ExecutionException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to seed Firestore stories", e);
        }
    }
    
    private static class StorySeed {
        private final String city;
        private final int year;
        private final GenderEnum gender;
        private final String text;
        private final String link;

        private StorySeed(String city, int year, GenderEnum gender, String text, String link) {
            this.city = city;
            this.year = year;
            this.gender = gender;
            this.text = text;
            this.link = link;
        }

        public String getCity() {
            return city;
        }

        public int getYear() {
            return year;
        }

        public GenderEnum getGender() {
            return gender;
        }

        public String getText() {
            return text;
        }

        public String getLink() {
            return link;
        }
    }
}
