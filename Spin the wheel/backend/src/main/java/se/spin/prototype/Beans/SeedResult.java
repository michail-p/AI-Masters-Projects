package se.spin.prototype.Beans;

public class SeedResult {
    private final String text;
    private final String link;

    public SeedResult(String text, String link) {
        this.text = text;
        this.link = link;
    }

    public String getText() {
        return text;
    }

    public String getLink() {
        return link;
    }
}
