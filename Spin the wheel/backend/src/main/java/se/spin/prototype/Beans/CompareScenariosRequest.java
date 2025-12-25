package se.spin.prototype.Beans;

public class CompareScenariosRequest {
    private SpinArguments spinArgumentsFirstStory;
    private SpinArguments spinArgumentsSecondStory;
    private GeneratedTextSources generatedTextSourcesFirstStory;
    private GeneratedTextSources generatedTextSourcesSecondStory;

    public SpinArguments getSpinArgumentsFirstStory() {
        return spinArgumentsFirstStory;
    }
    public void setSpinArgumentsFirstStory(SpinArguments spinArgumentsFirstStory) { this.spinArgumentsFirstStory = spinArgumentsFirstStory; }

    public SpinArguments getSpinArgumentsSecondStory() {
        return spinArgumentsSecondStory;
    }
    public void setSpinArgumentsSecondStory(SpinArguments spinArgumentsSecondStory) { this.spinArgumentsSecondStory = spinArgumentsSecondStory; }

    public GeneratedTextSources getGeneratedTextSourcesFirstStory() {
        return generatedTextSourcesFirstStory;
    }
    public void setGeneratedTextSourcesFirstStory(GeneratedTextSources generatedTextSourcesFirstStory) { this.generatedTextSourcesFirstStory = generatedTextSourcesFirstStory; }

    public GeneratedTextSources getGeneratedTextSourcesSecondStory() {
        return generatedTextSourcesSecondStory;
    }
    public void setGeneratedTextSourcesSecondStory(GeneratedTextSources generatedTextSourcesSecondStory) { this.generatedTextSourcesSecondStory = generatedTextSourcesSecondStory; }
}
