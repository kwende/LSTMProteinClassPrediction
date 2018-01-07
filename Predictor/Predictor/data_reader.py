import numpy as np

class Protein:
    
    def __init__(self, className, name, sequence, sequenceVector, classVector):
        self.ClassName = className
        self.Name = name
        self.Sequence = sequence

        self.SequenceVector = sequenceVector
        self.ClassVector = classVector


def read_annotated_fasta(filePath):
    ret = []

    with open(filePath) as file:
        lines = file.readlines()

        preProcessedData = []

        foldClassDictionary = {}
        foldClassCounter = 0

        moleculeDictionary = {}
        moleculeCounter = 0

        currentFoldClass = ""
        id = ""
        sequence = ""

        for line in lines:
            if line.startswith("-") or line.startswith("*") or line.startswith("\n"):
                continue
            elif line.startswith("TYPE"):

                # if we were working on a protein, save it.
                if not sequence is "":
                    preProcessedData.append((currentFoldClass, id, sequence))

                # pull out the fold class from the file
                currentFoldClass = line[(line.find(")") + 1):].strip()

                # maintain a dictionary of fold class to integer key.
                # this will be used to build our one-hot encoded vector
                # for class type
                if not currentFoldClass in foldClassDictionary:
                    foldClassDictionary[currentFoldClass] = foldClassCounter
                    foldClassCounter = foldClassCounter + 1

                # this is new data, so reset everything.
                id = ""
                sequence = ""
            elif line.startswith(">"):

                # if we were working on a protein, save it.
                if not sequence is "":
                    preProcessedData.append((currentFoldClass, id, sequence))

                # pull out the id of the protein
                id = line[1:].strip()

                # this is a new line, so reset the sequence.  class is the
                # same.
                sequence = ""
            else:

                # continue appending to the sequence data.
                line = line.strip()
                for c in line:
                    if not c in moleculeDictionary:
                        moleculeDictionary[c] = moleculeCounter
                        moleculeCounter = moleculeCounter + 1
                sequence += line

        vocabSize = moleculeCounter
        
        for tuple in preProcessedData:
            currentFoldClass = tuple[0]
            id = tuple[1]
            sequence = tuple[2]

            classVector = np.zeros([foldClassCounter])
            classVector[foldClassDictionary[currentFoldClass]] = 1

            sequenceVector = np.zeros([700,vocabSize])

            for a in range(0, len(sequence)):
                index = moleculeDictionary[sequence[a]]
                sequenceVector[a][index] = 1

            ret.append(Protein(currentFoldClass, id, sequence, sequenceVector, classVector))
    return ret