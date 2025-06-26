# Building OMR Project From Scratch

> Buiding the OMR Project as a standalone project from the beginning as the previous project has multiple dependencies.


## Phases
1. Work on an Annotation Tool
    * Make the Annotation tool as user friendly as possible.
    * Try to implement LabelImg if possible as its very simple and easy to use
    * **KNOWN ERRORS:** 
        * If the Resolution of the Image is higher then the Display then first there has to be a scroll function implemented.
        * Also, after Implementation of the Scroll Function, there has to be a perfect poiting system that can take the input of the clicks very easilt. The problem was, while the OMR was scrolled to annotate, then the coordintes it took was of the screen not the paper. Hence, points that coincide with each other after scolling the OMR results in Issues.
2. Testing The Annotation Tool
    * Test the annotation tool such that whwn required the tool can map the bounding boxes on the image as well.
3. *(High Level Step)* Implement the Way or WorkFlow to Extract the Answer marked by the Candidate.