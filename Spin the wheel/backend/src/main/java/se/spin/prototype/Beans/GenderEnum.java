package se.spin.prototype.Beans;

public enum GenderEnum {
    MALE("Male"),
    FEMALE("Female"),
    NONBINARY("Non-binary");

    private final String description;

    GenderEnum(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
