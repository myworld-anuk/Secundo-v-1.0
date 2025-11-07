import { Card } from "@/components/ui/card";
import { useEffect, useState } from "react";

interface ImagePreviewProps {
  file: File;
}

export const ImagePreview = ({ file }: ImagePreviewProps) => {
  const [previewUrl, setPreviewUrl] = useState<string>("");

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <Card className="overflow-hidden bg-card shadow-soft">
      <div className="p-6">
        <h3 className="text-lg font-semibold mb-4 text-card-foreground">Preview</h3>
        <div className="rounded-lg overflow-hidden border border-border bg-muted/20">
          <img
            src={previewUrl}
            alt="Sheet music preview"
            className="w-full h-auto"
          />
        </div>
      </div>
    </Card>
  );
};
