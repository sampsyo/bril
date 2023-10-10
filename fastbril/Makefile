TARGET_EXEC ?= fastbrili

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src
CONGIG_DiR ?= ./config
DOC_DIR ?= ./doc
SHELL = /bin/sh

CONFIGS := $(shell find ./config/ -name "*.cf")
GEN_HEAD := $(shell for FILE in $(CONFIGS); do echo $$FILE | awk -F'[/.]' '{printf "src/%s.h\n", $$4}'; done)
GEN_TEX := $(shell for FILE in $(CONFIGS); do echo $$FILE | awk -F'[/.]' '{printf "doc/%s.tex\n", $$4}'; done)

SRCS := $(shell find $(SRC_DIRS) -name *.c)

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))


CFLAGS += $(INC_FLAGS)

.PHONY: debug

debug: CFLAGS += -g -Og
debug: $(BUILD_DIR)/$(TARGET_EXEC)

.PHONY: coverage

coverage: CFLAGS += --coverage -DDEBUG -g3
coverage: $(GEN_HEAD)
	mkdir $(BUILD_DIR) && cd $(BUILD_DIR) && $(CC) $(CFLAGS) $(abspath $(SRCS)) -o $(TARGET_EXEC)

.PHONY: cov-report

cov-report:
	gcovr -r . --html --html-details -o cov-report/out.html

.PHONY: release

release: CFLAGS += -O3
release: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS) lib/lib.o
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# # assembly
# $(BUILD_DIR)/%.s.o: %.s
# 	$(MKDIR_P) $(dir $@)
# 	$(AS) $(ASFLAGS) -c $< -o $@

# c source
$(BUILD_DIR)/%.c.o: %.c | $(GEN_HEAD)
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# configured header files
$(GEN_HEAD): $(CONFIGS) srcgen.sh srcgen.awk
	./srcgen.sh $@

brb.pdf: $(GEN_TEX) $(DOC_DIR)/main.tex
	cd $(DOC_DIR) && latex -output-format=pdf main.tex && mv main.pdf brb.pdf

lib/lib.o: lib/lib.c
	cd lib && $(CC) -O3 -c lib.c

doc: brb.pdf

$(GEN_TEX): $(configs) docgen.sh docgen.awk
	$(MKDIR_P) $(dir $@)
	./docgen.sh $@

.PHONY: clean

clean:
	find . -name "*.aux" -o -name "*.log" -o -name "*.pdf" -o -name  "*~" -o \
	-name "*.gcda" -o -name "*.gcno" | xargs rm || true
	$(RM) $(GEN_HEAD)
	$(RM) $(GEN_TEX)
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
