from music21 import stream, note, meter, key

s = stream.Score()
part = stream.Part()

# Add a time signature and key signature
part.append(meter.TimeSignature('4/4'))
part.append(key.Key('C'))

# Add a few notes
for pitch in ["C4", "D4", "E4", "F4"]:
    n = note.Note(pitch)
    n.quarterLength = 1
    part.append(n)

s.append(part)

# Export to MusicXML file
s.write('musicxml', fp='simple_melody.xml')
print("Exported MusicXML to simple_melody.xml")