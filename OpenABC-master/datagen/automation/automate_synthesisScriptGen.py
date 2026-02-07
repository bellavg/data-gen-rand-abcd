import os
from typing import Optional
import argparse

homeDir = None#os.environ["HOME"]
srcFolder = None #sys.argv[1]
# Src folder is a 'design' folder under abcScripts from where the scripts to be copied.
graphDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","bench")
scriptsDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","synScripts")
libraryCellFolder = None #os.path.join(homeDir,"OPENABC_DATASET","lib")
libraryFile = None # Full path to the library file


numSynthesizedScript = 1500
delimiter = "\n"
designSet6 = ['128','256','512','1024','2048','4096','8192','16384']
designs = designSet6


## Perform all the optimization steps for all the designs

def genSynthesisScripts():
    for i in range(numSynthesizedScript):
        if srcFolder is None:
            raise ValueError("Source folder not properly initialized")
        srcFile = os.path.join(srcFolder, 'abc' + str(i) + '.script')
        origScriptFile = open(srcFile, 'r', encoding='utf-8')
        fileLines = origScriptFile.readlines()
        origScriptFile.close()
        for des in designs:
            if scriptsDataFolder is None or graphDataFolder is None:
                raise ValueError("Required folders not properly initialized")
            scriptFolder = os.path.join(scriptsDataFolder, des)
            if(not os.path.exists(scriptFolder)):
                os.mkdir(scriptFolder)
            graphDumpFolder = os.path.join(graphDataFolder, des)
            scriptFilePath = os.path.join(scriptFolder, 'abc' + str(i) + '.script')
            scriptFile = open(scriptFilePath, 'w+', encoding='utf-8')
            
            # Create metadata directory and CSV file if needed (once per design)
            if graphDataFolder is None:
                raise ValueError("Graph data folder not properly initialized")
            metadataFolder = os.path.join(graphDataFolder, des, 'metadata')
            if not os.path.exists(metadataFolder):
                os.makedirs(metadataFolder, exist_ok=True)
            
            csvFile = os.path.join(metadataFolder, f'{des}.csv')
            if not os.path.exists(csvFile):
                with open(csvFile, 'w', encoding='utf-8') as f:
                    f.write("file_path,design,recipe_id,step_id,tier_id,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout\n")
            
            if libraryFile is None:
                raise ValueError("Library file not specified")
            readLibLine = "read " + libraryFile + delimiter
            scriptFile.write(readLibLine)
            fileLines[1] = "read "+graphDumpFolder+os.sep+des+"_orig.aig"+delimiter
            scriptFile.write(fileLines[1])
            scriptFile.write("strash"+delimiter)
            
            # Write stats to log (system command removed - will collect metadata post-synthesis)
            scriptFile.write(f'print_stats{delimiter}')
            
            firstPathFileName = os.path.join(graphDumpFolder, "syn" + str(i),des + "_syn" + str(i) + "_step1.aig")
            dumpFirstGraphLine = "write " + firstPathFileName + delimiter
            scriptFile.write(dumpFirstGraphLine)
            
            numSteps = 1
            for line in fileLines[2:-8]:
                scriptFile.write(line)
                
                # Increment step counter
                numSteps += 1
                
                # Write stats to log (system command removed - will collect metadata post-synthesis)
                scriptFile.write(f'print_stats{delimiter}')
                
                intermediatePathFileName = os.path.join(graphDumpFolder,"syn"+str(i),des+"_syn"+str(i)+"_step"+str(numSteps)+".aig")
                dumpIntermediateGraphLine = "write " + intermediatePathFileName + delimiter
                scriptFile.write(dumpIntermediateGraphLine)
            
            # Final step: write final stats to log (system command removed - will collect metadata post-synthesis)
            scriptFile.write(f'print_stats{delimiter}')
            
            scriptFile.close()


def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir, srcFolder, graphDataFolder, scriptsDataFolder, libraryCellFolder, libraryFile
    # Initialize global variables with proper types
    homeDir = cmdArgs.home
    srcFolder = cmdArgs.script
    if not (os.path.exists(homeDir) and os.path.exists(srcFolder)):
        print("\nPlease rerun with appropriate paths")
    graphDataFolder = os.path.join(homeDir,"OPENABC_DATASET","bench")
    scriptsDataFolder = os.path.join(homeDir,"OPENABC_DATASET","synScripts")
    libraryCellFolder = os.path.join(homeDir,"OPENABC_DATASET","lib")
    
    # Set library file: use provided path or default to nangate45.lib
    if cmdArgs.lib:
        libraryFile = cmdArgs.lib
    else:
        libraryFile = os.path.join(libraryCellFolder,"nangate45.lib")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='SYNTHESIS RECIPE GENERATOR', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--home',required=True, help="OpenABC dataset home path")
    parser.add_argument('--script', required=True, help="Sample script folder path of 1500 synthesis scripts")
    parser.add_argument('--lib', required=False, help="Path to library file (default: OPENABC_DATASET/lib/nangate45.lib)")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    genSynthesisScripts()

if __name__ == '__main__':
    main()
