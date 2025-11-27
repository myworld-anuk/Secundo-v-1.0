// src/components/ConversionActions.tsx
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ConversionActionsProps {
  hasFile: boolean;
  isLoading: boolean;
  onConvert: () => void;
  xml: string | null;
  downloadUrl: string | null;
}

export const ConversionActions = ({
  hasFile,
  isLoading,
  onConvert,
  xml,
  downloadUrl,
}: ConversionActionsProps) => {
  return (
    <div className="space-y-4">
      <Card className="p-4 shadow-soft">
        <h3 className="font-semibold mb-2 text-card-foreground">
          Convert to MusicXML
        </h3>
        <p className="text-sm text-muted-foreground mb-4">
          Click convert to run the OMR model on your uploaded measure.
        </p>
        <Button
          onClick={onConvert}
          disabled={!hasFile || isLoading}
          className="w-full"
        >
          {isLoading ? "Converting…" : "Convert to MusicXML"}
        </Button>
      </Card>

      {xml && (
        <Card className="p-4 shadow-soft">
          <h4 className="font-semibold mb-2 text-card-foreground">
            MusicXML Preview
          </h4>
          <pre className="text-xs max-h-48 overflow-auto whitespace-pre-wrap bg-muted p-2 rounded-md">
            {xml}
          </pre>
        </Card>
      )}

      {downloadUrl && (
        <a
          href={downloadUrl}
          download="measure.musicxml"
          className="inline-flex items-center justify-center w-full px-4 py-2 rounded-lg bg-secondary text-secondary-foreground text-sm font-semibold shadow"
        >
          Download MusicXML
        </a>
      )}
    </div>
  );
};