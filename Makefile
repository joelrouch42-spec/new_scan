CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2

# Main trading system
TARGET = trading_system
SOURCES = trading_system.c nasdaq_symbols.c ibkr_connector.c data_manager.c settings_parser.c
HEADERS = trading_system.h nasdaq_symbols.h ibkr_connector.h data_manager.h settings_parser.h

# Test modules
NASDAQ_TARGET = test_nasdaq
NASDAQ_SOURCES = nasdaq_symbols.c test_nasdaq.c
NASDAQ_HEADERS = nasdaq_symbols.h


# Build main target
$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

# Build NASDAQ test
$(NASDAQ_TARGET): $(NASDAQ_SOURCES) $(NASDAQ_HEADERS)
	$(CC) $(CFLAGS) -o $(NASDAQ_TARGET) $(NASDAQ_SOURCES)

# Run the main program
run: $(TARGET)
	./$(TARGET)

# Test NASDAQ symbols
test-nasdaq: $(NASDAQ_TARGET)
	./$(NASDAQ_TARGET)

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(NASDAQ_TARGET)

# Install dependencies (placeholder)
deps:
	@echo "Installing C dependencies..."
	@echo "TODO: Install cjson, libcurl, etc."

.PHONY: run debug clean deps test-nasdaq