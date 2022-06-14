#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>

void myinit(void) {
    // currently, does nothing
}

void mykeyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // ESCAPE
            exit(0);
            break;
    }
}

void mydisplay(void) {
    glClearColor(1.0F, 1.0F, 1.0F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();
}

void myreshape(int w, int h)
{
    printf("reshaped (%d, %d)\n", w, h);
}

void mymouse(int button, int state, int x, int y)
{
    printf("mouse:button: %d, state: %d, (%d, %d)\n", button, state, x, y);
}

void mymenu(int id)
{
    switch(id) {
        case 1: printf("A is selected\n");
        break;
        case 2: printf("B is selected\n");
        break;
    }
}

int main(int argc, char* argv[])
{
    int menu_id;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("simple");
    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);
    glutMouseFunc(mymouse);
    glutReshapeFunc(myreshape);

    menu_id = glutCreateMenu(mymenu);
    glutAddMenuEntry("select A", 1);
    glutAddMenuEntry("select B", 2);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}