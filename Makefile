.PHONY: build install clean

build:
	@echo "Building the project..."
	mkdir -p build
	cd build && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
	cd build && ninja

# Copies the custom python package to the active virtualenv:
install:
	@echo "Installing .so to active virtualenv..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ No virtualenv active! Please activate one."; \
		exit 1; \
	fi; \
	sofile="$$(find build -name 'pyfluid*.so' | head -n 1)"; \
	target_dir="$$(find "$$VIRTUAL_ENV/lib" -type d -name "site-packages" | head -n 1)"; \
	if [ -z "$$sofile" ] || [ -z "$$target_dir" ]; then \
		echo "❌ Could not locate .so or site-packages"; exit 1; \
	fi; \
	cp "$$sofile" "$$target_dir/pyfluid.so"; \
	echo "✅ Installed $$sofile to $$target_dir"
		
clean:
	rm -rf build