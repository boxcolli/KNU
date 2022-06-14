<실행 방법>----------------------------------------



>>
$./run_triangle.sh
기존의 triangle 파일을 빠르게 실행하기 위한 스크립트입니다.



>>
$./run.sh <SUT file> <case file> <driver file>
다른 이름의 파일을 입력하기 위한 스크립트입니다.



>>
$./run_literal.sh <SUT file> <case file> <driver file>
테스트 케이스 파일에 리터럴 표현이 없을 경우, 데이터 타입을 유추하여 리터럴 표현을 덧붙입니다.
float, double, long double의 경우는 변화가 없습니다.)










<파일 설명>----------------------------------------



Makefile
	driver 생성 프로그램을 컴파일합니다.



global.h
classes.cpp
main.cpp
	driver 생성 프로그램 소스코드입니다.
	-l 옵션을 주면 테스트 케이스를 리터럴 처리합니다.
	-r 옵션은 기본 옵션으로, 테스트 케이스를 문자열 그대로 처리합니다.
