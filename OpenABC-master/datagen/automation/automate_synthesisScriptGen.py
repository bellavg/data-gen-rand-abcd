import os, sys,random,shutil
import argparse

homeDir = None#os.environ["HOME"]
srcFolder = None #sys.argv[1]
# Src folder is a 'design' folder under abcScripts from where the scripts to be copied.
graphDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","bench")
scriptsDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","synScripts")
libraryCellFolder = None #os.path.join(homeDir,"OPENABC_DATASET","lib")


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
            readLibLine = "read "+os.path.join(libraryCellFolder,"nangate45.lib")+delimiter
            scriptFile.write(readLibLine)
            fileLines[1] = "read_aig "+graphDumpFolder+os.sep+des+"_orig.aig"+delimiter
            scriptFile.write(fileLines[1])
            scriptFile.write("strash"+delimiter)
            firstPathFileName = os.path.join(graphDumpFolder, "syn" + str(i),des + "_syn" + str(i) + "_step0.aig"+delimiter)
            dumpFirstGraphLine = "write_aig " + firstPathFileName
            scriptFile.write(dumpFirstGraphLine)
            numSteps = 1
            for line in fileLines[2:-8]:
                scriptFile.write(line)
                intermediatePathFileName = os.path.join(graphDumpFolder,"syn"+str(i),des+"_syn"+str(i)+"_step"+str(numSteps)+".aig"+delimiter)
                dumpIntermediateGraphLine = "write_aig " + intermediatePathFileName
                scriptFile.write(dumpIntermediateGraphLine)
                numSteps+=1
            scriptFile.write("map -B 0.9"+delimiter+"topo"+delimiter+"stime -c"+delimiter)
            scriptFile.close()


def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir, srcFolder, graphDataFolder,scriptsDataFolder,libraryCellFolder
    homeDir = cmdArgs.home
    srcFolder = cmdArgs.script
    if not (os.path.exists(homeDir) and os.path.exists(srcFolder)):
        print("\nPlease rerun with appropriate paths")
    graphDataFolder = os.path.join(homeDir,"OPENABC_DATASET","bench")
    scriptsDataFolder = os.path.join(homeDir,"OPENABC_DATASET","synScripts")
    libraryCellFolder = os.path.join(homeDir,"OPENABC_DATASET","lib")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='SYNTHESIS RECIPE GENERATOR', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--home',required=True, help="OpenABC dataset home path")
    parser.add_argument('--script', required=True, help="Sample script folder path of 1500 synthesis scripts")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    genSynthesisScripts()

if __name__ == '__main__':
    main()
