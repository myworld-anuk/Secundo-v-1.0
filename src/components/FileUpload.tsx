import { useState, useCallback } from "react";
import { Upload, Music, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

export const FileUpload = ({ onFileSelect, selectedFile, onClear }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files && files[0]) {
        handleFile(files[0]);
      }
    },
    []
  );

  const handleFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file");
      return;
    }
    
    if (!file.type.includes("png")) {
      toast.error("Please upload a PNG file");
      return;
    }

    onFileSelect(file);
    toast.success("Sheet music uploaded successfully!");
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  return (
    <Card className="relative overflow-hidden border-2 border-dashed transition-all duration-300 bg-card shadow-soft hover:shadow-elegant">
      {selectedFile ? (
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Music className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="font-medium text-card-foreground">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClear}
              className="hover:bg-destructive/10 hover:text-destructive"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>
      ) : (
        <div
          className={`p-12 text-center transition-all duration-300 ${
            isDragging ? "bg-primary/5 border-primary" : ""
          }`}
          onDragEnter={handleDragIn}
          onDragLeave={handleDragOut}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 rounded-2xl bg-gradient-primary">
              <Upload className="w-8 h-8 text-primary-foreground" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-card-foreground mb-2">
                Upload Sheet Music
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                Drag and drop your PNG file here, or click to browse
              </p>
            </div>
            <label htmlFor="file-upload">
              <Button variant="default" className="cursor-pointer" asChild>
                <span>Choose File</span>
              </Button>
              <input
                id="file-upload"
                type="file"
                accept="image/png"
                onChange={handleFileInput}
                className="hidden"
              />
            </label>
          </div>
        </div>
      )}
    </Card>
  );
};
