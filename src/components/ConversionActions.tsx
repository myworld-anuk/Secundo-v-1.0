import { Download, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";

interface ConversionActionsProps {
  hasFile: boolean;
}

export const ConversionActions = ({ hasFile }: ConversionActionsProps) => {
  const handleConvert = () => {
    toast.info("Conversion logic will be implemented here");
  };

  const handleDownload = () => {
    toast.info("Download functionality will be available after conversion");
  };

  return (
    <Card className="bg-gradient-primary p-1 shadow-elegant">
      <div className="bg-card rounded-lg p-6">
        <div className="flex flex-col gap-4">
          <div>
            <h3 className="text-lg font-semibold text-card-foreground mb-2">
              Convert to MusicXML
            </h3>
            <p className="text-sm text-muted-foreground">
              Once conversion logic is immplemented, this section will process the sheet music
              and generate MusicXML output.
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              onClick={handleConvert}
              disabled={!hasFile}
              className="flex-1 bg-gradient-primary hover:opacity-90 transition-opacity"
            >
              <Sparkles className="w-4 h-4 mr-2" />
              Convert
            </Button>
            <Button
              onClick={handleDownload}
              disabled={!hasFile}
              variant="outline"
              className="flex-1"
            >
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};
