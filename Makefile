run:
	@go run main.go
build:
	@go build -o bin/exam main.go
build-win:
	 @GOOS=windows GOARCH=amd64 go build -o bin/exam.exe main.go
run-prod:
	@ ./bin/exam
