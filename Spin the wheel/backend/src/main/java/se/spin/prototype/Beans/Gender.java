package se.spin.prototype.Beans;

public class Gender {
    private GenderEnum id;
    private String description;

    // Default constructor needed for Jackson deserialization
    public Gender() {}

    public Gender(GenderEnum id, String description) {
        this.id = id;
        this.description = description;
    }

    public GenderEnum getId() { return id; }
    public void setId(GenderEnum id) { this.id = id; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}
