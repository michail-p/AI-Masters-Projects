package se.spin.prototype.Controllers;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import se.spin.prototype.Beans.Gender;
import se.spin.prototype.Beans.GenderEnum;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@RestController
@RequestMapping("/api/parameterization")
public class ParameterizationController {

    @GetMapping("/genders")
    public List<Gender> getGenders() {
        return Arrays.stream(GenderEnum.values()).map(g -> new Gender(g, g.getDescription())).toList();
    }

    @GetMapping("/times")
    public List<Integer> getTimes() {
        
        int start = 1800;
        int end = 2000;
        List<Integer> decades = new ArrayList<>();

        for (int year = start; year <= end; year += 50) {
            decades.add(year);
        }

        if ((end % 50) != 0) {
            decades.add(end);
        }

        return decades;
    }

    @GetMapping("/places")
    public List<String> getPlaces() {
        return List.of(
                "Stockholm",
                "Göteborg",
                "Malmö"
        );
    }
}
