# Building OMR Project From Scratch

> Buiding the OMR Project as a standalone project from the beginning as the previous project has multiple dependencies.


## Phases

Section 1: **ANNOTATION TOOL** 
1. Work on an Annotation Tool _(LabelImg for now)_
    * Make the Annotation tool as user friendly as possible.
    * Try to implement LabelImg if possible as its very simple and easy to use
    * **KNOWN ERRORS:** 
        * If the Resolution of the Image is higher then the Display then first there has to be a scroll function implemented.
        * Also, after Implementation of the Scroll Function, there has to be a perfect poiting system that can take the input of the clicks very easilt. The problem was, while the OMR was scrolled to annotate, then the coordintes it took was of the screen not the paper. Hence, points that coincide with each other after scolling the OMR results in Issues.
2. Testing The Annotation Tool
    * Test the annotation tool such that whwn required the tool can map the bounding boxes on the image as well.
3. *(High Level Step)* Implement the Way or WorkFlow to Extract the Answer marked by the Candidate.

<br>

Section 2: <ins>**MAIN JOB**</ins>
1. Detect the **Anchors** on the OMR sucessfully.
2. After Successful Anchor Detection, Find the **Skew Angle**.
   * Pair consecutive anchors to detect the anchor
   * Set a Skew Threshold, which of exceeded, place the Image inside **Warning** Folder.


<br>

**NOTES**:
1. LabelImg was sucessfully setup on 27th June
2. Anchor Detection and Skewness was completed on 30th June.
   * Testing on **Different Type of OMR** left. 
   * Test Results for **Different Types of OMR**, it was found that, for Square Anchors, it was working fine, but with OMR having **Circular Anchors**, it was performing Badly.
   * Currently working on the Pipeline with **Square Anchors**.
3. **Anchor Detection testing for Square Anchors** was successfully coompleted on 01st June *(upon the provided data)*
4. Working on Successfull Detection of **Question and Answers**.