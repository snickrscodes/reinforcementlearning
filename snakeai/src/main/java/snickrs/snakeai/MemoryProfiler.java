package snickrs.snakeai;

public class MemoryProfiler {

    Runtime rt;

    long usedMB;
    
    public MemoryProfiler(){
        rt = Runtime.getRuntime();
    }
    
    public void printMemoryUsage() {
    	usedMB = (rt.totalMemory() - rt.freeMemory()) / 1024 / 1024;
    	System.out.println("Memory Used: " + usedMB + "MB");
    }

}