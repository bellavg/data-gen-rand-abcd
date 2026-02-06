import os, sys,random,shutil
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
        srcFile = os.path.join(srcFolder,'abc'+str(i)+'.script')
        origScriptFile = open(srcFile,'r')
        fileLines = origScriptFile.readlines()
        origScriptFile.close()
        for des in designs:
            scriptFolder = os.path.join(scriptsDataFolder,des)
            if(not os.path.exists(scriptFolder)):
                os.mkdir(scriptFolder)
            graphDumpFolder = os.path.join(graphDataFolder,des)
            scriptFilePath = os.path.join(scriptFolder, 'abc' + str(i) + '.script')
            scriptFile = open(scriptFilePath, 'w+')
            
            # Create metadata directory and CSV file if needed (once per design)
            metadataFolder = os.path.join(graphDataFolder, des, 'metadata')
            if not os.path.exists(metadataFolder):
                os.makedirs(metadataFolder, exist_ok=True)
            
            csvFile = os.path.join(metadataFolder, f'{des}.csv')
            if not os.path.exists(csvFile):
                with open(csvFile, 'w') as f:
                    f.write("file_path,design,recipe_id,step_id,tier_id,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout\n")
            
            readLibLine = "read "+libraryFile+delimiter
            scriptFile.write(readLibLine)
            fileLines[1] = "read "+graphDumpFolder+os.sep+des+"_orig.aig"+delimiter
            scriptFile.write(fileLines[1])
            scriptFile.write("strash"+delimiter)
            
            # Capture and write initial stats to temp file, then collect metadata
            temp_stats_file = os.path.join(graphDumpFolder, f"temp_stats_syn{i}_step1.txt")
            scriptFile.write(f'print_stats > {temp_stats_file}{delimiter}')
            metadata_script = os.path.join(homeDir, "dataset_tools", "metadata_collector.py")
            scriptFile.write(f'system "python3 {metadata_script} {des} {i} 1 {graphDumpFolder} {temp_stats_file}"{delimiter}')
            
            firstPathFileName = os.path.join(graphDumpFolder, "syn" + str(i),des + "_syn" + str(i) + "_step1.aig")
            dumpFirstGraphLine = "write " + firstPathFileName + delimiter
            scriptFile.write(dumpFirstGraphLine)
            
            numSteps = 1
            for line in fileLines[2:-8]:
                scriptFile.write(line)
                
                # Increment step counter
                numSteps += 1
                
                # Capture and write stats to temp file, then collect metadata
                temp_stats_file = os.path.join(graphDumpFolder, f"temp_stats_syn{i}_step{numSteps}.txt")
                scriptFile.write(f'print_stats > {temp_stats_file}{delimiter}')
                scriptFile.write(f'system "python3 {metadata_script} {des} {i} {numSteps} {graphDumpFolder} {temp_stats_file}"{delimiter}')
                
                intermediatePathFileName = os.path.join(graphDumpFolder,"syn"+str(i),des+"_syn"+str(i)+"_step"+str(numSteps)+".aig")
                dumpIntermediateGraphLine = "write " + intermediatePathFileName + delimiter
                scriptFile.write(dumpIntermediateGraphLine)
            
            # Final step: capture final logical statistics to temp file
            temp_stats_file = os.path.join(graphDumpFolder, f"temp_stats_syn{i}_step21.txt")
            scriptFile.write(f'print_stats > {temp_stats_file}{delimiter}')
            scriptFile.write(f'system "python3 {metadata_script} {des} {i} 21 {graphDumpFolder} {temp_stats_file}"{delimiter}')
            
            scriptFile.close()


def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir, srcFolder, graphDataFolder,scriptsDataFolder,libraryCellFolder,libraryFile
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
