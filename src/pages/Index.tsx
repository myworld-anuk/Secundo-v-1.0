import { useState } from "react";
import { Music2 } from "lucide-react";
import { FileUpload } from "@/components/FileUpload";
import { ImagePreview } from "@/components/ImagePreview";
import { ConversionActions } from "@/components/ConversionActions";

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleClear = () => {
    setSelectedFile(null);
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10 shadow-soft">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-primary">
              <Music2 className="w-6 h-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">Sheet Music Converter</h1>
              <p className="text-sm text-muted-foreground">PNG to MusicXML</p>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-12">
        <div className="max-w-3xl mx-auto text-center mb-12">
          <h2 className="text-4xl font-bold text-foreground mb-4">
            Transform Sheet Music to{" "}
            <span className="bg-gradient-primary bg-clip-text text-transparent">
              Digital Format
            </span>
          </h2>
          <p className="text-lg text-muted-foreground">
            Upload your sheet music images and convert them to MusicXML format for easy editing and
            sharing with notation software.
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto space-y-6">
          <FileUpload
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClear={handleClear}
          />

          {selectedFile && (
            <div className="grid md:grid-cols-2 gap-6 animate-in fade-in duration-500">
              <ImagePreview file={selectedFile} />
              <ConversionActions hasFile={!!selectedFile} />
            </div>
          )}
        </div>

        {/* Features */}
        {!selectedFile && (
          <div className="max-w-4xl mx-auto mt-16 grid md:grid-cols-3 gap-6">
            <div className="text-center p-6 rounded-xl bg-card shadow-soft">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <Music2 className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-card-foreground mb-2">Easy Upload</h3>
              <p className="text-sm text-muted-foreground">
                Simply drag and drop your PNG sheet music files
              </p>
            </div>
            <div className="text-center p-6 rounded-xl bg-card shadow-soft">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-accent/10 flex items-center justify-center">
                <Music2 className="w-6 h-6 text-accent" />
              </div>
              <h3 className="font-semibold text-card-foreground mb-2">Smart Conversion</h3>
              <p className="text-sm text-muted-foreground">
                Advanced processing to recognize musical notation
              </p>
            </div>
            <div className="text-center p-6 rounded-xl bg-card shadow-soft">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <Music2 className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-card-foreground mb-2">MusicXML Output</h3>
              <p className="text-sm text-muted-foreground">
                Download ready-to-use MusicXML files
              </p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export default Index;
